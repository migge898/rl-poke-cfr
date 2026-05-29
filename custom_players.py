import numpy as np
import pickle
import os
import datetime
from collections import defaultdict
from poke_env.player import Player
from poke_env.data import GenData
from helper import StateHelper

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
                self.q_table = defaultdict(lambda: np.zeros(6), loaded_dict)
            print(f"Loaded Q-Table with {len(self.q_table)} states.")