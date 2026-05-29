import asyncio
import random
from poke_env.player import RandomPlayer
from poke_env.teambuilder import Teambuilder
from custom_players import MaxDamagePlayer, TabularQLearningPlayer

# Your list of pokemon strings
TEAM_LIST = [
"""Starmie @ Life Orb
Ability: Natural Cure
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Surf
- Ice Beam
- Thunderbolt
- Psychic""",

"""Heatran @ Choice Scarf
Ability: Flash Fire
EVs: 252 SpA / 4 SpD / 252 Spe
Naive Nature
- Fire Blast
- Earth Power
- Flash Cannon
- Dragon Pulse""",

"""Scizor @ Choice Band
Ability: Technician
EVs: 248 HP / 252 Atk / 8 SpD
Adamant Nature
- Bullet Punch
- U-turn
- Superpower
- Pursuit"""
]

class ShuffledTeambuilder(Teambuilder):
    def __init__(self, team_list):
        self.converted_teams = [self.join_team(self.parse_showdown_team(t)) for t in team_list]

    def yield_team(self):
        shuffled = self.converted_teams.copy()
        random.shuffle(shuffled)
        return shuffled[0]
    
async def main():
    concurrent_battles = 10
    
    # Initialize the teambuilder
    teambuilder = ShuffledTeambuilder(TEAM_LIST)

    # Setup Players
    random_player = RandomPlayer(
        battle_format="gen4ou", 
        team=teambuilder, 
        max_concurrent_battles=concurrent_battles
    )
    max_damage_player = MaxDamagePlayer(
        battle_format="gen4ou", 
        team=teambuilder, 
        max_concurrent_battles=concurrent_battles
    )

    q_learning_player = TabularQLearningPlayer(
        battle_format="gen4ou", 
        team=teambuilder, 
        max_concurrent_battles=concurrent_battles,
        q_table_path="q_table_gen4.pkl",
        save_replays=False
    )

    # 1. Training Phase
    n_train_per_player = 1000
    if n_train_per_player > 0:
        # Split the training count
        print(f"Training for {n_train_per_player} battles against RandomPlayer ...")

        await q_learning_player.battle_against(random_player, n_battles=n_train_per_player)

        print(f"Training for {n_train_per_player} battles against MaxDamagePlayer ...")
        await q_learning_player.battle_against(max_damage_player, n_battles=n_train_per_player)

        print(f"Training finished. Final Epsilon: {q_learning_player.epsilon:.4f}")
        q_learning_player.save_q_table() 

    # 2. Evaluation Phase
    q_wins_before = q_learning_player.n_won_battles
    n_eval = 100
    q_learning_player.epsilon = 0.0 # No exploration
    
    print(f"Evaluation against Random for {n_eval} battles...")
    await q_learning_player.battle_against(random_player, n_battles=n_eval)
    
    eval_wins = q_learning_player.n_won_battles - q_wins_before
    win_rate = (eval_wins / n_eval) * 100
    print(f"Win Rate in Evaluation: {win_rate:>.2f}%")
    print(f"States in Q-Table: {len(q_learning_player.q_table)}")

if __name__ == "__main__":
    asyncio.run(main())