import asyncio
from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer

async def main():
    # 1. Configure the bot to connect to your local server
    # Since we used --no-security, the password doesn't matter
    bot_account = AccountConfiguration("MyLocalBot", "password")
    
    # 2. Initialize the RandomPlayer
    bot = RandomPlayer(
        account_configuration=bot_account,
    )

    print("Bot is online! Waiting for challenges...")

    # 3. Tell the bot to accept any challenge it receives
    # It will stay active until you stop the script
    await bot.accept_challenges(None, n_challenges=10)

if __name__ == "__main__":
    asyncio.run(main())