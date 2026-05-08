import numpy as np
import pickle
import os
import datetime
from collections import defaultdict
from poke_env.player import Player
from poke_env.data import GenData
from torch.utils.tensorboard import SummaryWriter

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
        self.q_table = defaultdict(lambda: np.zeros(6)) # 4 moves + 2 switches
        self.load_q_table()

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = "logs/q_agent_" + current_time
        # Tensorboard Writer
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"Tensorboard logs will be saved to {log_dir}.")
        
        self.battle_count = 0
        
        # Hyperparameter
        self.epsilon = 0.2
        self.alpha = 0.1
        self.gamma = 0.9
        
        self.last_state = None
        self.last_action = None

    def _get_state(self, battle):
        return (
            battle.active_pokemon.species,
            battle.opponent_active_pokemon.species,
            int(battle.active_pokemon.current_hp_fraction * 5),
            int(battle.opponent_active_pokemon.current_hp_fraction * 5)
        )

    def choose_move(self, battle):
        current_state = self._get_state(battle)
        
        if self.last_state is not None:
            self._update_q(self.last_state, self.last_action, 0.1, current_state)

        available_indices = [i for i, _ in enumerate(battle.available_moves)]
        available_indices += [i + 4 for i, _ in enumerate(battle.available_switches)]

        if not available_indices:
            return self.choose_random_move(battle)

        if np.random.random() < self.epsilon:
            action_idx = np.random.choice(available_indices)
        else:
            q_values = self.q_table[current_state]
            mask = np.full(6, -np.inf)
            mask[available_indices] = 0
            action_idx = np.argmax(q_values + mask)

        self.last_state = current_state
        self.last_action = action_idx

        if action_idx < 4:
            return self.create_order(battle.available_moves[action_idx])
        else:
            return self.create_order(battle.available_switches[action_idx - 4])

    def _update_q(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state]) if next_state is not None else 0
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

    def on_battle_end(self, battle):
        reward = 10.0 if battle.won else -10.0
        self._update_q(self.last_state, self.last_action, reward, None)
        
        # Logging to Tensorboard
        self.battle_count += 1
        self.writer.add_scalar("Battle/Reward", reward, self.battle_count)
        self.writer.add_scalar("Battle/Cumulative_Wins", self.n_won_battles, self.battle_count)
        self.writer.add_scalar("Stats/Q_Table_Size", len(self.q_table), self.battle_count)
        self.writer.flush()
        self.last_state = None

    def save_q_table(self):
        with open(self.q_table_path, "wb") as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-Table saved to {self.q_table_path}")

    def load_q_table(self):
        if os.path.exists(self.q_table_path):
            with open(self.q_table_path, "rb") as f:
                loaded_dict = pickle.load(f)
                self.q_table = defaultdict(lambda: np.zeros(6), loaded_dict)
            print(f"Loaded Q-Table with {len(self.q_table)} states.")
        else:
            print("No existing Q-Table found. Starting fresh.")