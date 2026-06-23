import asyncio
from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from stable_baselines3 import PPO

from custom_players import WangPlayer
from env import MaskedActorCriticPolicy, WangEnv
from teams import TEST_TEAM

async def main():
    bot_account = AccountConfiguration("MyLocalBot", "password")
    
    bot = WangPlayer(
        policy=None,
        account_configuration=bot_account,
        battle_format="gen4ou",
        max_concurrent_battles=1,
        team=TEST_TEAM
    )

    # bot = RandomPlayer(
    #     account_configuration=bot_account,
    # )

    print("Bot is online! Waiting for challenges...")

    await bot.accept_challenges(None, n_challenges=10)

if __name__ == "__main__":
    asyncio.run(main())
    # test