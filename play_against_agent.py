import asyncio
from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from stable_baselines3 import PPO

from custom_players import WangPlayer

def load_model_policy(model_path: str):
    """
    Loads a PPO model and returns its policy.
    """
    model = PPO.load(model_path)
    return model.policy

async def main():
    bot_account = AccountConfiguration("MyLocalBot", "password")
    checkpoint_path = "checkpoints/ppo_gen4_selfplay_v1_37000000_steps.zip"
    
    bot = WangPlayer(
        policy=load_model_policy(checkpoint_path),
        account_configuration=bot_account,
        battle_format="gen4randombattle",
        max_concurrent_battles=1,
    )

    print("Bot is online! Waiting for challenges...")

    await bot.accept_challenges(None, n_challenges=10)

if __name__ == "__main__":
    asyncio.run(main())