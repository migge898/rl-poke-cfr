from typing import Awaitable

import numpy as np
import pickle
import os
import datetime
from collections import defaultdict
from poke_env import to_id_str
from poke_env.battle import AbstractBattle, Effect, Field, PokemonType, SideCondition, Status, Weather
from poke_env.environment import SinglesEnv
from poke_env.player import BattleOrder, DefaultBattleOrder, Player
from poke_env.data import GenData
import torch
from helper import VOLATILE_EFFECTS_LIST, StateHelper, bin_pp, item_to_deterministic_float, load_abilities_from_gen_data, one_hot

class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            best_move = max(
                battle.available_moves,
                key=lambda move: self._calculate_score(move, battle)
            )
            return self.create_order(best_move)
        return self.choose_random_move(battle)

    def _calculate_score(self, move, battle):
        bp = move.base_power
        if battle.active_pokemon.ability == "technician" and bp <= 60:
            bp *= 1.5
        if move.type in [battle.active_pokemon.type_1, battle.active_pokemon.type_2]:
            bp *= 1.5
        effectiveness = move.type.damage_multiplier(
            battle.opponent_active_pokemon.type_1,
            battle.opponent_active_pokemon.type_2,
            type_chart=GenData.from_gen(battle.gen).type_chart,
        )
        return bp * effectiveness

class TabularQLearningPlayer(Player):
    def __init__(self, *args, q_table_path="q_table.pkl", **kwargs):
        super().__init__(*args, **kwargs)
        self.q_table_path = q_table_path
        
        self.state_helper = StateHelper()
        
        # 4 move slots + 3 switch slots = 7 fixed actions
        self.q_table = defaultdict(lambda: np.zeros(7)) 
        self.load_q_table()

        # Hyperparameters
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.997
        
        self.alpha = 0.01
        self.alpha_min = 0.001
        self.alpha_decay = 0.999

        self.gamma = 0.95

        self.last_states = {}
        self.last_action_indices = {}
        self.last_faint_counts = {}

        self.team_ordered_names = None


    def _compute_reward(self, battle, end_of_battle=False):
        tag = battle.battle_tag
        
        # Current counts
        current_my_fainted = sum(1 for p in battle.team.values() if p.fainted)
        current_opp_fainted = sum(1 for p in battle.opponent_team.values() if p.fainted)
        
        # Initialize if this is the first turn
        if tag not in self.last_faint_counts:
            self.last_faint_counts[tag] = (0, 0)
            
        last_my, last_opp = self.last_faint_counts[tag]
        
        # Calculate differences
        faints_inflicted = current_opp_fainted - last_opp
        faints_suffered = current_my_fainted - last_my
        
        reward = (faints_inflicted * 10.0) - (faints_suffered * 10.0)
        
        # Win/Loss bonus at the end
        if end_of_battle:
            if battle.won:
                reward += 20.0
            else:
                reward -= 20.0
                
        # Update the tracker
        self.last_faint_counts[tag] = (current_my_fainted, current_opp_fainted)
        return reward

    def _get_team_list(self, battle):
        """
        Returns a fixed list of species names in the order they appear in the team.
        """
        if self.team_ordered_names is None:
            self.team_ordered_names = sorted([p.species for p in battle.team.values()])
        return self.team_ordered_names

    def _get_action_mapping(self, battle):
        """
        Maps available actions to FIXED slots 0-6.
        0-3: Moves (based on the position in the active pokemon's moveset)
        4-6: Switches (4=Team Member 0, 5=Team Member 1, 6=Team Member 2)
        """
        mapping = {}
        team_list = self._get_team_list(battle)
        
        # --- 1. Map Moves (Slots 0-3) ---
        active_pokemon = battle.active_pokemon
        known_moves = list(active_pokemon.moves.values())
        
        for move in battle.available_moves:
            if move in known_moves:
                slot = known_moves.index(move)
                if slot < 4:
                    mapping[slot] = move

        # --- 2. Map Switches (Slots 4-6) ---
        # Instead of taking the first available, we check which team member 
        # is available and put it in its specific slot.
        for switch_option in battle.available_switches:
            # Find the index of this specific pokemon in our fixed team list
            if switch_option.species in team_list:
                team_idx = team_list.index(switch_option.species)
                slot = 4 + team_idx
                mapping[slot] = switch_option
                
        return mapping

    def choose_move(self, battle):
        tag = battle.battle_tag
        current_state = self._get_state(battle)
        action_mapping = self._get_action_mapping(battle)
        available_indices = list(action_mapping.keys())

        # Q-Learning Update (Standard SARSA-like or Q-Learning update)
        if tag in self.last_states:
            # For the update, we need to know which actions are possible in the NEXT state
            # to calculate the max Q correctly.
            step_reward = self._compute_reward(battle, end_of_battle=False)
            
            self._update_q(
                self.last_states[tag],
                self.last_action_indices[tag],
                step_reward,
                current_state,
                available_indices
            )

        # If no actions are available (e.g. during animations or weird state), fallback
        if not available_indices:
            return self.choose_random_move(battle)

        # Action Selection (Epsilon-Greedy)
        if np.random.random() < self.epsilon:
            action_idx = np.random.choice(available_indices)
        else:
            q_values = self.q_table[current_state]
            # Create a mask: -infinity for unavailable actions, 0 for available ones
            mask = np.full(7, -np.inf)
            mask[available_indices] = 0
            # Argmax will now only pick from available indices
            action_idx = int(np.argmax(q_values + mask))

        self.last_states[tag] = current_state
        self.last_action_indices[tag] = action_idx

        return self.create_order(action_mapping[action_idx])

    def _update_q(self, state, action, reward, next_state, next_available_indices=None):
        """
        Updates the Q-table. 
        next_available_indices ensures we only consider valid moves for the max_next_q.
        """
        if action is None:
            return
            
        current_q = self.q_table[state][action]
        
        if next_state is None:
            # Terminal state (battle ended)
            max_next_q = 0
        else:
            if next_available_indices:
                # Mask out invalid actions in the next state for the update
                next_q_values = self.q_table[next_state]
                max_next_q = np.max(next_q_values[next_available_indices])
            else:
                max_next_q = np.max(self.q_table[next_state])
                
        # Q-Learning Formula
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

    def _battle_finished_callback(self, battle):
        tag = battle.battle_tag

        final_reward = self._compute_reward(battle, end_of_battle=True)

        last_state = self.last_states.get(tag)
        last_action_idx = self.last_action_indices.get(tag)

        self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)

        if self.epsilon != 0.0: 
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        if last_state is not None:
            self._update_q(last_state, last_action_idx, final_reward, None)
        
        # Reset trackers for next battle
        if tag in self.last_states: del self.last_states[tag]
        if tag in self.last_action_indices: del self.last_action_indices[tag]
        if tag in self.last_faint_counts: del self.last_faint_counts[tag]

    def _get_state(self, battle):
        # Placeholder for your StateHelper
        return self.state_helper.build_state(battle)

    def save_q_table(self):
        with open(self.q_table_path, "wb") as f:
            pickle.dump(dict(self.q_table), f)
        # print(f"Saved Q-Table. States: {len(self.q_table)}")
        # # print states of q table
        # for state in self.q_table:
        #     print(f"State: {state}")


    def load_q_table(self):
        if os.path.exists(self.q_table_path):
            with open(self.q_table_path, "rb") as f:
                loaded_dict = pickle.load(f)
                self.q_table = defaultdict(lambda: np.zeros(7), loaded_dict)
            print(f"Loaded Q-Table with {len(self.q_table)} states.")

class WangPlayer(Player):
    # --- CLASS ATTRIBUTES (Accessible by static methods) ---
    # We initialize these once. Subprocesses will inherit them.
    _gen_data = GenData.from_gen(4)
    
    # Pre-calculate Dictionaries for O(1) lookup speed
    SPECIES_TO_IDX = {s: i for i, s in enumerate(sorted(list(_gen_data.pokedex.keys())))}
    MOVES_TO_IDX = {m: i for i, m in enumerate(sorted(list(_gen_data.moves.keys())))}
    
    # Abilities need special handling (as you had it)
    _abilities = load_abilities_from_gen_data(_gen_data)
    ABILITIES_TO_IDX = {a: i for i, a in enumerate(_abilities)}
    
    # Types
    _types = [t.name for t in PokemonType if t not in [PokemonType.THREE_QUESTION_MARKS, PokemonType.STELLAR]]
    TYPE_TO_IDX = {t: i for i, t in enumerate(_types)}
    
    VOLATILE_LIST = VOLATILE_EFFECTS_LIST
    def __init__(self, policy=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy
    
    @staticmethod
    def embed_battle(battle: AbstractBattle, debug: bool = False) -> np.ndarray:
        features = [] # total length of 4001
        if debug:
            debug_dict = {}

        # --- 1. FIELD FEATURES (Table A.1) ---
        for w in [Weather.SUNNYDAY, Weather.RAINDANCE, Weather.HAIL, Weather.SANDSTORM]:
            duration = battle.weather.get(w, 0)
            val = 8 if duration is True else (duration if isinstance(duration, int) else 0)
            vec = one_hot(val, 0, 8)
            features.append(vec)
            if debug:
                debug_dict[f"field_{w.name.lower()}"] = vec.tolist()

        features.append([1.0 if not battle.weather else 0.0])
        if debug:
            debug_dict["field_no_weather"] = [1.0 if not battle.weather else 0.0]
        
        tr_turns = battle.fields.get(Field.TRICK_ROOM, 0)
        tr_val = 6 if tr_turns is False else (tr_turns if isinstance(tr_turns, int) else 0)
        features.append(one_hot(tr_val, 0, 6))
        if debug:
            debug_dict["field_trick_room"] = one_hot(tr_val, 0, 6).tolist()

        # --- Force Switch Logic ---

        # Me: Am I prompted to switch?
        me_forced = 1 if battle.force_switch else 0

        # Representation as two separate Length-2 One-Hots (Total 4 features)
        features.append(one_hot(me_forced, 0, 1))

        if debug:
            debug_dict["force_switch_me"] = one_hot(me_forced, 0, 1).tolist()

        # # unknown: Number of unrevealed opponent Pokemon (Length 7: 0-6)
        unknown_count = 6 - len(battle.opponent_team)
        features.append(one_hot(unknown_count, 0, 6))
        if debug:
            debug_dict["global_unknown_count"] = unknown_count

        # --- 2. SIDE CONDITIONS (Table A.1) ---
        sides = [battle.side_conditions, battle.opponent_side_conditions]
        side_names = ["me", "opp"]

        # Stealth Rock (Length 2: Me then Opponent)
        for i, s in enumerate(sides):
            sr = 1 if SideCondition.STEALTH_ROCK in s else 0
            features.append(one_hot(sr, 0, 1))
            if debug:
                debug_dict[f"side_{side_names[i]}_stealth_rock"] = sr
            
        # Spikes (Length 4 each: Me then Opponent)
        for i, s in enumerate(sides):
            spikes = s.get(SideCondition.SPIKES, 0)
            features.append(one_hot(spikes, 0, 3))
            if debug:
                debug_dict[f"side_{side_names[i]}_spikes"] = spikes
            
        # Toxic Spikes (Length 3 each: Me then Opponent)
        for i, s in enumerate(sides):
            t_spikes = s.get(SideCondition.TOXIC_SPIKES, 0)
            features.append(one_hot(t_spikes, 0, 2))
            if debug:
                debug_dict[f"side_{side_names[i]}_toxic_spikes"] = t_spikes

        # Reflect, Light Screen, Safeguard (Grouped by feature, then Me then Opponent)
        for sc, m_v in [(SideCondition.REFLECT, 9), (SideCondition.LIGHT_SCREEN, 9), (SideCondition.SAFEGUARD, 6)]:
            for i, s in enumerate(sides):
                t = s.get(sc, 0)
                # If True (unknown duration), guess middle value. Else use int or 0.
                v = (m_v // 2) if t is True else (t if isinstance(t, int) else 0)
                vec = one_hot(v, 0, m_v)
                features.append(vec)
                if debug:
                    debug_dict[f"side_{side_names[i]}_{sc.name.lower()}"] = vec.tolist()

        # --- 3. POKEMON FEATURES (Table A.2) ---
        if debug:
            debug_dict["pokemon"] = []
        my_team = list(battle.team.values())
        opp_team = list(battle.opponent_team.values())

        for i in range(12):
            mon = my_team[i] if i < 6 else (opp_team[i-6] if (i-6) < len(opp_team) else None)
            mon_vec = []
            if debug:
                mon_debug = {}
            
            is_unknown = 1 if (i >= 6 and mon is None) else 0
            if debug:
                mon_debug["is_unknown"] = bool(is_unknown)
                mon_debug["slot"] = i

            if mon is None or is_unknown:
                padding = np.zeros(323, dtype=np.float32) # due to the fixed size of the feature vector for a Pokemon
                padding[-1] = 1.0 if is_unknown else 0.0
                features.append(padding)
                if debug:
                    debug_dict["pokemon"].append(mon_debug)
                continue

            # Species, Ability, Item

            s_idx = WangPlayer.SPECIES_TO_IDX.get(mon.species, 0)
            mon_vec.append([s_idx / len(WangPlayer.SPECIES_TO_IDX)])
            
            a_id = to_id_str(mon.ability) if mon.ability else ""
            a_idx = WangPlayer.ABILITIES_TO_IDX.get(a_id, 0)
            mon_vec.append([a_idx / len(WangPlayer.ABILITIES_TO_IDX)])
            
            mon_vec.append([item_to_deterministic_float(mon.item)])

            # Moves & PP
            if debug:
                mon_debug["moves"] = {}
            moves = list(mon.moves.values())
            for m_i in range(4):
                move = moves[m_i] if m_i < len(moves) else None
                move_idx = WangPlayer.MOVES_TO_IDX.get(move.id, 0) if move else 0
                mon_vec.append([move_idx / len(WangPlayer.MOVES_TO_IDX)])
                mon_vec.append(one_hot(bin_pp(move.current_pp) if move else 0, 0, 3))
                if debug and move:
                    mon_debug["moves"][move.id] = move.current_pp

            # Last Move
            last_move = mon.last_move
            if debug:
                mon_debug["last_move"] = last_move.id if last_move else None
            last_move_idx = WangPlayer.MOVES_TO_IDX.get(last_move.id, 0) if last_move else 0
            mon_vec.append([last_move_idx / len(WangPlayer.MOVES_TO_IDX)])

            # Types
            if debug:
                mon_debug["types"] = [mon.type_1, mon.type_2]
            type_vec = np.zeros(18, dtype=np.float32)
            if mon.type_1:
                idx1 = WangPlayer.TYPE_TO_IDX.get(mon.type_1.name, 0)
                type_vec[idx1] = 1.0
            if mon.type_2:
                idx2 = WangPlayer.TYPE_TO_IDX.get(mon.type_2.name, 0)
                type_vec[idx2] = 1.0
            mon_vec.append(type_vec)

            # HP Fraction
            if debug:
                mon_debug["hp_fraction"] = mon.current_hp_fraction
            hp_bin = 0 if mon.current_hp == 0 else int(mon.current_hp_fraction * 16) + 1
            mon_vec.append(one_hot(hp_bin, 0, 17))

            # Stat Boosts
            if debug:
                mon_debug["boosts"] = {}
            for s in ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]:
                boost_val = int(mon.boosts.get(s, 0))
                if debug:
                    mon_debug["boosts"][s] = boost_val
                mon_vec.append(one_hot(boost_val, -6, 6))

            # Volatile Effects
            if debug:
                mon_debug["volatiles"] = [effect for effect in WangPlayer.VOLATILE_LIST if effect in mon.effects]
            for effect in WangPlayer.VOLATILE_LIST:
                active = 1 if effect in mon.effects else 0
                mon_vec.append(one_hot(active, 0, 1))

            # Specific Counters
            if debug:
                mon_debug["counters"] = {
                    "encore": mon.effects.get(Effect.ENCORE, 0),
                    "taunt": mon.effects.get(Effect.TAUNT, 0),
                    "magnet_rise": mon.effects.get(Effect.MAGNET_RISE, 0),
                    "slow_start": mon.effects.get(Effect.SLOW_START, 0)
                }
            mon_vec.append(one_hot(mon.effects.get(Effect.ENCORE, 0), 0, 8))
            mon_vec.append(one_hot(mon.effects.get(Effect.TAUNT, 0), 0, 5))
            mon_vec.append(one_hot(mon.effects.get(Effect.MAGNET_RISE, 0), 0, 6))
            mon_vec.append(one_hot(mon.effects.get(Effect.SLOW_START, 0), 0, 5))

            # Gender
            gender_map = {"NEUTRAL": 2, "MALE": 0, "FEMALE": 1}
            if debug:
                mon_debug["gender"] = str(mon.gender)
            mon_vec.append(one_hot(gender_map.get(str(mon.gender), 2), 0, 2))

            # Status
            if debug:
                mon_debug["status"] = str(mon.status) if mon.status else ("FNT" if mon.current_hp == 0 else None)
            status_vec = np.zeros(7, dtype=np.float32)
            if mon.current_hp == 0:
                status_vec[6] = 1.0
            elif mon.status is not None:
                status_mapping = {Status.BRN: 0, Status.FRZ: 1, Status.PAR: 2, Status.PSN: 3, Status.SLP: 4, Status.TOX: 5}
                idx = status_mapping.get(mon.status)
                if idx is not None: status_vec[idx] = 1.0
            mon_vec.append(status_vec)

            # Status Counters
            if debug:
                mon_debug["toxic_counter"] = mon.status_counter if mon.status == Status.TOX else 0
                mon_debug["sleep_counter"] = mon.status_counter if mon.status == Status.SLP else 0
            mon_vec.append(one_hot(mon.effects.get(Status.TOX, 0), 0, 20))
            mon_vec.append(one_hot(mon.effects.get(Status.SLP, 0), 0, 10))

            # Size
            mon_vec.append(one_hot(int(np.log10(mon.weight)) if mon.weight > 0 else 0, 0, 4))
            mon_vec.append(one_hot(int(np.log10(mon.height * 100)) if mon.height > 0 else 0, 0, 3))

            # Combat Mechanics
            if debug:
                mon_debug["first_turn"] = bool(mon.first_turn)
                mon_debug["protect_counter"] = mon.effects.get(Effect.PROTECT, 0)
                mon_debug["must_recharge"] = bool(mon.must_recharge)
                mon_debug["preparing"] = bool(mon.preparing)

            mon_vec.append(one_hot(1 if mon.first_turn else 0, 0, 1))
            mon_vec.append(one_hot(mon.effects.get(Effect.PROTECT, 0), 0, 5))
            mon_vec.append(one_hot(1 if mon.must_recharge else 0, 0, 1))
            mon_vec.append(one_hot(1 if mon.preparing else 0, 0, 1))

            # Active & Team
            if debug:
                mon_debug["active"] = bool(mon.active)
                mon_debug["is_opponent"] = i >= 6
            mon_vec.append(one_hot(1 if mon.active else 0, 0, 1))
            mon_vec.append(one_hot(1 if i >= 6 else 0, 0, 1))

            # unknown flag
            mon_vec.append([0.0])

            features.append(np.concatenate(mon_vec, dtype=np.float32))
            if debug:
                debug_dict["pokemon"].append(mon_debug)
        feature_vector = np.concatenate(features, dtype=np.float32)
        if debug:
            print(f"Feature vector length: {len(feature_vector)}")
        return feature_vector

    def choose_move(self, battle: AbstractBattle) -> BattleOrder | Awaitable[BattleOrder]:
        if battle.wait: return DefaultBattleOrder()
        obs = self.embed_battle(battle)
        mask = np.array(SinglesEnv.get_action_mask(battle))

        if self.policy is None:
            # Fallback to random if no policy is provided
            return self.choose_random_move(battle)
        
        with torch.no_grad():
            obs_dict = {
                "observation": torch.as_tensor(obs, device=self.policy.device).unsqueeze(0),
                "action_mask": torch.as_tensor(mask, device=self.policy.device).unsqueeze(0),
            }
            action, _, _ = self.policy.forward(obs_dict)
        action_idx = action.cpu().numpy()[0]
        return SinglesEnv.action_to_order(action_idx, battle)