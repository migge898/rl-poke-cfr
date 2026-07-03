import os
import multiprocessing

# FIX 1: Disable macOS fork safety crash
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import asyncio
import matplotlib
import torch
import numpy as np
from datetime import datetime

# Prevent GUI issues on macOS
matplotlib.use('Agg') 

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from poke_env import Player, RandomPlayer, SimpleHeuristicsPlayer
from custom_players import MaxDamagePlayer, WangPlayer
from env import WangEnv, MaskedActorCriticPolicy

# --- Configuration ---
BATTLE_FORMAT = "gen4randombattle"
MODEL_NAME = "wang_ppo_gen4_model"
LOG_DIR = "./tensorboard_logs/"
CHECKPOINT_DIR = "./checkpoints/"

WANG_PARAMS = {
    "learning_rate": 10**-4.23,
    "n_epochs": 7,
    "gamma": 0.9999,
    "gae_lambda": 0.754,
    "ent_coef": 0.0588,
    "vf_coef": 0.4375,
    "max_grad_norm": 0.5430,
    "batch_size": 1024,
}

def train():

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    num_envs = 4
    # Create the environments
    env = SubprocVecEnv([WangEnv.create_env for _ in range(num_envs)], start_method="spawn")

    # Choose device mps > cuda > cpu
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # if cpu has less overhead use it instead
    device = "cpu"
    print(f"Training on: {device}")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 10000 // num_envs),
        save_path=CHECKPOINT_DIR,
        name_prefix=MODEL_NAME
    )

    model = PPO(
        MaskedActorCriticPolicy,
        env,
        device=device,
        verbose=1,
        tensorboard_log=LOG_DIR,
        n_steps=3072 // num_envs,
        **WANG_PARAMS
    )

    debug_timesteps = 5000 
    print(f"Starting debug training for {debug_timesteps} steps...")
    
    model.learn(
        total_timesteps=debug_timesteps,
        progress_bar=True,
        callback=checkpoint_callback,
        tb_log_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    model.save(MODEL_NAME)
    print(f"Final model saved.")
    env.close()

    # Evaluation needs its own asyncio loop
    print("\nStarting evaluation...")
    asyncio.run(evaluate_model(model.policy))

async def evaluate_model(policy):
    agent = WangPlayer(
        policy=policy, 
        battle_format=BATTLE_FORMAT, 
        max_concurrent_battles=10
    )
    opponents = [
        RandomPlayer(battle_format=BATTLE_FORMAT, max_concurrent_battles=10),
        MaxDamagePlayer(battle_format=BATTLE_FORMAT, max_concurrent_battles=10),
        SimpleHeuristicsPlayer(battle_format=BATTLE_FORMAT, max_concurrent_battles=10)
    ]
    
    for opp in opponents:
        await agent.battle_against(opp, n_battles=20)
        win_rate = (agent.n_won_battles / agent.n_finished_battles) * 100
        print(f"Win rate vs {opp.username}: {win_rate:.1f}%")
        agent.reset_battles()

if __name__ == "__main__":
    train()