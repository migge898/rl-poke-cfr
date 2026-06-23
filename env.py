from poke_env.battle import AbstractBattle
from poke_env.environment import SinglesEnv
import torch
import numpy as np

from gymnasium.spaces import Box, Dict as GymDict

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

from custom_players import WangPlayer

N_FEATURES = 1337

class WangFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=N_FEATURES)

    def forward(self, obs):
        return obs["observation"]

class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            net_arch=[256, 256, 256], # Wang uses 3-layer MLP with 256 dim (A.0.2)
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
        self.observation_space = GymDict({
            "observation": Box(-1, 10, shape=(N_FEATURES,), dtype=np.float32),
            "action_mask": Box(0, 1, shape=(22,), dtype=np.float32) # Standard Singles action space
        })

    def embed_battle(self, battle: AbstractBattle):
        # We reuse the embedding logic from WangPlayer
        if not hasattr(self, 'embedder'):
            self.embedder = WangPlayer(battle_format="gen4ou")
        return self.embedder.embed_battle(battle)

    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(battle, victory_value=1.0)