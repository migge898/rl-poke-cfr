from poke_env import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.battle import AbstractBattle
from poke_env.environment import SingleAgentWrapper, SinglesEnv
import torch
import numpy as np

from gymnasium.spaces import Box, Dict as GymDict

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.monitor import Monitor
from torch import nn

from custom_players import WangPlayer

N_FEATURES = 4001


class WangFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=256)

        self.trunk = nn.Sequential(
            nn.Linear(N_FEATURES, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

    def forward(self, obs):
        features = obs["observation"]
        return self.trunk(features)

class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        wang_net_arch = dict(
            pi=[256, 256],  # 2-layer MLP Actor head
            vf=[256, 256]   # 2-layer MLP Critic head
        )

        super().__init__(
            *args,
            **kwargs,
            net_arch=wang_net_arch,
            activation_fn=torch.nn.ReLU,
            features_extractor_class=WangFeaturesExtractor,
        )

    def _get_action_dist_from_latent(self, latent_pi):
        action_logits = self.action_net(latent_pi)
        mask = torch.where(self._mask == 1, 0, float("-inf"))
        return self.action_dist.proba_distribution(action_logits + mask)

    def forward(self, obs, deterministic=False):
        self._mask = obs["action_mask"]
        return super().forward(obs, deterministic)

    def evaluate_actions(self, obs, actions):
        self._mask = obs["action_mask"]
        return super().evaluate_actions(obs, actions)
    
class WangEnv(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_spaces = {
            agent: Box(-1, 4, shape=(N_FEATURES,), dtype=np.float32)
            for agent in self.possible_agents
        }
    
    @classmethod
    def create_env(cls) -> Monitor:
        env = cls(battle_format="gen4randombattle", log_level=40, open_timeout=None)
        opponent = SimpleHeuristicsPlayer(start_listening=False)
        return Monitor(SingleAgentWrapper(env, opponent))

    def embed_battle(self, battle: AbstractBattle):
        return WangPlayer.embed_battle(battle)

    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle,
            fainted_value=2.0,
            hp_value=1.0,
            victory_value=30.0
        )