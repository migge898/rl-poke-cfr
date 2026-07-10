import os
import asyncio
import torch
import numpy as np
from datetime import datetime

# FIX: macOS multiprocessing safety
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
import matplotlib
matplotlib.use('Agg') 

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from poke_env import RandomPlayer, SimpleHeuristicsPlayer
from custom_players import MaxDamagePlayer, WangPlayer
from env import WangEnv, MaskedActorCriticPolicy

# --- Configuration ---
BATTLE_FORMAT = "gen4randombattle"
MODEL_NAME = "ppo_gen4_selfplay_v1"
LOG_DIR = "./tensorboard_logs/"
CHECKPOINT_DIR = "./checkpoints/"

WANG_PARAMS = {
    "learning_rate": 10**-4.0,
    "n_epochs": 7,
    "gamma": 0.9999,
    "gae_lambda": 0.754,
    "ent_coef": 0.0588,
    "vf_coef": 0.4375,
    "max_grad_norm": 0.5430,
    "batch_size": 1024,
    "clip_range": 0.0829,
    "clip_range_vf": 0.0184,
}

# --- Wang Validation Callback ---
class WangValidationCallback(BaseCallback):
    """
    Validates the current policy against a heuristic opponent (SimpleHeuristicsPlayer) every `check_freq` steps.
    Logs the win rate to TensorBoard.
    """
    def __init__(self, check_freq: int, n_battles: int = 200, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.n_battles = n_battles

    def _on_step(self) -> bool:
        # check_freq bezieht sich auf globale Timesteps
        if self.num_timesteps % self.check_freq == 0:
            
            win_rate = asyncio.run(self.evaluate_performance())
            
            self.logger.record("eval/win_rate_vs_heuristic", win_rate)
            
        return True

    async def evaluate_performance(self):
        test_agent = WangPlayer(
            policy=self.model.policy,
            battle_format=BATTLE_FORMAT,
            max_concurrent_battles=20
        )
        opponent = SimpleHeuristicsPlayer(battle_format=BATTLE_FORMAT, max_concurrent_battles=20)
        
        await test_agent.battle_against(opponent, n_battles=self.n_battles)
        win_rate = (test_agent.n_won_battles / test_agent.n_finished_battles) * 100
        return win_rate

def train():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    num_envs = 10 
    
    base_env = WangEnv(battle_format=BATTLE_FORMAT, log_level=40)
    
    env = ss.pettingzoo_env_to_vec_env_v1(base_env)
    
    env = ss.concat_vec_envs_v1(
        env, 
        num_vec_envs=num_envs, 
        num_cpus=num_envs, 
        base_class="stable_baselines3"
    )

    device = "cpu"
    print(f"Training selfplay on: {device}")

    wang_eval_callback = WangValidationCallback(check_freq=20000, n_battles=200)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 1_000_000 // (num_envs * 2)),
        save_path=CHECKPOINT_DIR,
        name_prefix=MODEL_NAME
    )

    # 4. PPO Modell
    model = PPO(
        MaskedActorCriticPolicy,
        env,
        device=device,
        verbose=1,
        tensorboard_log=LOG_DIR,
        n_steps=1024,
        **WANG_PARAMS
    )

    total_steps = 800 *  60000
    
    model.learn(
        total_timesteps=total_steps,
        callback=[checkpoint_callback, wang_eval_callback],
        progress_bar=True,
        tb_log_name="run_selfplay"
    )

    model.save(MODEL_NAME)
    env.close()

if __name__ == "__main__":
    train()