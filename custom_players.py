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
        
        # Initialize the helper
        self.state_helper = StateHelper()
        
        # 4 moves + 2 possible switch targets = 6 actions
        self.q_table = defaultdict(lambda: np.zeros(6)) 
        self.load_q_table()

        # Hyperparameters
        self.epsilon = 0.2
        self.alpha = 0.1
        self.gamma = 0.9
        
        self.last_state = None
        self.last_action_idx = None

    def _get_state(self, battle):
        """Uses the helper to build the complex state tuple."""
        return self.state_helper.build_state(battle)

    def _get_action_mapping(self, battle):
        """Maps available actions to fixed slots 0-5."""
        mapping = {}
        
        # Map moves to slots 0-3 based on their position in the moveset
        current_moves = list(battle.active_pokemon.moves.values())
        for move in battle.available_moves:
            if move in current_moves:
                slot = current_moves.index(move)
                mapping[slot] = move
        
        # Map switches to slots 4-8 based on team position
        team_list = list(battle.team.values())
        for switch_out in battle.available_switches:
            if switch_out in team_list:
                slot = 4 + team_list.index(switch_out)
                if slot <= 5:
                    mapping[slot] = switch_out
        return mapping

    def choose_move(self, battle):
        current_state = self._get_state(battle)
        action_mapping = self._get_action_mapping(battle)
        available_indices = list(action_mapping.keys())

        # --- Debug Start ---
        # print(f"\n--- Turn {battle.turn} ---")
        # print(f"Available indices: {available_indices}")
        # for idx, action in action_mapping.items():
        #    print(f"  Slot {idx}: {action}")
        # --- Debug Ende ---

        # Q-Learning Update
        if self.last_state is not None:
            self._update_q(self.last_state, self.last_action_idx, 0.0, current_state)

        if not available_indices:
            return self.choose_random_move(battle)

        # Action Selection (Epsilon-Greedy)
        if np.random.random() < self.epsilon:
            action_idx = np.random.choice(available_indices)
        else:
            q_values = self.q_table[current_state]
            mask = np.full(6, -np.inf)
            mask[available_indices] = 0
            action_idx = np.argmax(q_values + mask)

        self.last_state = current_state
        self.last_action_idx = action_idx

        # Use the mapping to return the actual move/switch object
        return self.create_order(action_mapping[action_idx])

    def _update_q(self, state, action, reward, next_state):
        if action is None:
            return
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state]) if next_state is not None else 0
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

    def on_battle_end(self, battle):
        reward = 10.0 if battle.won else -10.0
        self._update_q(self.last_state, self.last_action_idx, reward, None)
        self.last_state = None
        self.last_action_idx = None

    def save_q_table(self):
        with open(self.q_table_path, "wb") as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-Table saved to {self.q_table_path}")
        print(len(self.q_table))
        print(self.q_table)

    def load_q_table(self):
        if os.path.exists(self.q_table_path):
            with open(self.q_table_path, "rb") as f:
                loaded_dict = pickle.load(f)
                self.q_table = defaultdict(lambda: np.zeros(6), loaded_dict)
            print(f"Loaded Q-Table with {len(self.q_table)} states.")
            