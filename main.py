import asyncio
from poke_env.player import RandomPlayer
from custom_players import MaxDamagePlayer, TabularQLearningPlayer

GEN_4_TEAM = """
Starmie @ Life Orb
Ability: Natural Cure
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Surf
- Ice Beam
- Thunderbolt
- Psychic

Heatran @ Choice Scarf
Ability: Flash Fire
EVs: 252 SpA / 4 SpD / 252 Spe
Naive Nature
- Fire Blast
- Earth Power
- Flash Cannon
- Dragon Pulse

Scizor @ Choice Band
Ability: Technician
EVs: 248 HP / 252 Atk / 8 SpD
Adamant Nature
- Bullet Punch
- U-turn
- Superpower
- Pursuit
"""

async def main():
    concurrent_battles = 10
    # Setup Players
    random_player = RandomPlayer(battle_format="gen4ou", team=GEN_4_TEAM, max_concurrent_battles=concurrent_battles)
    max_damage_player = MaxDamagePlayer(battle_format="gen4ou", team=GEN_4_TEAM, max_concurrent_battles=concurrent_battles)
    
    # Q-Player will automatically try to load 'q_table.pkl'
    q_learning_player = TabularQLearningPlayer(
        battle_format="gen4ou", 
        team=GEN_4_TEAM, 
        max_concurrent_battles=concurrent_battles,
        q_table_path="q_table_gen4.pkl",
        save_replays=False
    )

    # 1. Training Phase (Skip if you think your table is good enough)
    n_train = 1
    if n_train > 0:
        print(f"Training for {n_train} battles...")
        await q_learning_player.battle_against(max_damage_player, n_battles=n_train)
        q_learning_player.save_q_table() # Save after training

    exit()
    # 2. Evaluation Phase
    q_wins_before = q_learning_player.n_won_battles
    n_eval = 20
    q_learning_player.epsilon = 0.0 # Evaluation mode: no exploration
    
    print(f"Evaluation against Random for {n_eval} battles...")
    await q_learning_player.battle_against(random_player, n_battles=n_eval)
    
    eval_wins = q_learning_player.n_won_battles - q_wins_before
    print(f"Win Rate in Evaluation: {(eval_wins/n_eval)*100:>.2f}%")

if __name__ == "__main__":
    asyncio.run(main())