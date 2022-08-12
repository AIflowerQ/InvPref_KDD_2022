import copy
import math

from torch import nn
import torch
import numpy as np

from functions import ReverseLayerF


class InvariantPreferenceLearner(nn.Module):
    def __init__(self):
        super(InvariantPreferenceLearner, self).__init__()

    def forward(self, *embed):
        raise NotImplementedError

    def get_L2_reg(self) -> torch.Tensor:
        raise NotImplementedError

    def get_L1_reg(self) -> torch.Tensor:
        raise NotImplementedError


class EnvAwarePreferenceLearner(nn.Module):
    def __init__(self):
        super(EnvAwarePreferenceLearner, self).__init__()

    def forward(self, *embed):
        raise NotImplementedError

    def get_L2_reg(self) -> torch.Tensor:
        raise NotImplementedError

    def get_L1_reg(self) -> torch.Tensor:
        raise NotImplementedError


class ScorePredictor(nn.Module):
    def __init__(self):
        super(ScorePredictor, self).__init__()

    def forward(self, *preferences):
        raise NotImplementedError

    def get_L2_reg(self) -> torch.Tensor:
        raise NotImplementedError

    def get_L1_reg(self) -> torch.Tensor:
        raise NotImplementedError


class EnvClassifier(nn.Module):
    def __init__(self):
        super(EnvClassifier, self).__init__()

    def forward(self, invariant_preferences):
        raise NotImplementedError

    def get_L2_reg(self) -> torch.Tensor:
        raise NotImplementedError

    def get_L1_reg(self) -> torch.Tensor:
        raise NotImplementedError


class BasicRecommender(nn.Module):
    def __init__(self, user_num: int, item_num: int):
        super(BasicRecommender, self).__init__()
        self.user_num: int = user_num
        self.item_num: int = item_num

    def forward(self, *args):
        raise NotImplementedError

    def get_L2_reg(self, *args) -> torch.Tensor:
        raise NotImplementedError

    def get_L1_reg(self, *args) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, user_id) -> torch.Tensor:
        raise NotImplementedError


class BasicExplicitRecommender(nn.Module):
    def __init__(self, user_num: int, item_num: int):
        super(BasicExplicitRecommender, self).__init__()
        self.user_num: int = user_num
        self.item_num: int = item_num

    def forward(self, *args):
        raise NotImplementedError

    def get_L2_reg(self, *args) -> torch.Tensor:
        raise NotImplementedError

    def get_L1_reg(self, *args) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, user_id, item_id) -> torch.Tensor:
        raise NotImplementedError


class GeneralDebiasImplicitRecommender(BasicRecommender):
    def __init__(self, user_num: int, item_num: int, env_num: int):
        super(GeneralDebiasImplicitRecommender, self).__init__(
            user_num=user_num, item_num=item_num
        )
        self.env_num: int = env_num

    def forward(self, *args):
        raise NotImplementedError

    def get_L2_reg(self, *args) -> torch.Tensor:
        raise NotImplementedError

    def get_L1_reg(self, *args) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, user_id) -> torch.Tensor:
        raise NotImplementedError

    def cluster_predict(self, *args) -> torch.Tensor:
        raise NotImplementedError


class GeneralDebiasExplicitRecommender(BasicExplicitRecommender):
    def __init__(self, user_num: int, item_num: int, env_num: int):
        super(GeneralDebiasExplicitRecommender, self).__init__(
            user_num=user_num, item_num=item_num
        )
        self.env_num: int = env_num

    def forward(self, *args):
        raise NotImplementedError

    def get_L2_reg(self, *args) -> torch.Tensor:
        raise NotImplementedError

    def get_L1_reg(self, *args) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, user_id, item_id) -> torch.Tensor:
        raise NotImplementedError

    def cluster_predict(self, *args) -> torch.Tensor:
        raise NotImplementedError


class InnerProductLinearTransInvariantPreferenceLearner(InvariantPreferenceLearner):
    def __init__(self, factor_dim: int):
        super(InnerProductLinearTransInvariantPreferenceLearner, self).__init__()
        self.linear_map = nn.Linear(factor_dim, factor_dim)
        self._init_weight()
        self.elements_num: float = float(factor_dim * factor_dim)
        self.bias_num: float = float(factor_dim)

    def forward(self, users_embed, items_embed):
        return self.linear_map(users_embed * items_embed)

    def get_L1_reg(self) -> torch.Tensor:
        return torch.norm(self.linear_map.weight, 1) / self.elements_num \
               + torch.norm(self.linear_map.bias, 1) / self.bias_num

    def get_L2_reg(self) -> torch.Tensor:
        return torch.norm(self.linear_map.weight, 2).pow(2) / self.elements_num \
               + torch.norm(self.linear_map.bias, 2).pow(2) / self.bias_num

    def _init_weight(self):
        torch.nn.init.xavier_uniform_(self.linear_map.weight)


class InnerProductLinearTransEnvAwarePreferenceLearner(EnvAwarePreferenceLearner):
    def __init__(self, factor_dim: int):
        super(InnerProductLinearTransEnvAwarePreferenceLearner, self).__init__()
        self.linear_map = nn.Linear(factor_dim, factor_dim)
        self._init_weight()
        self.elements_num: float = float(factor_dim * factor_dim)
        self.bias_num: float = float(factor_dim)

    def forward(self, users_embed, items_embed, ens_embed):
        return self.linear_map(users_embed * items_embed * ens_embed)

    def get_L1_reg(self) -> torch.Tensor:
        return torch.norm(self.linear_map.weight, 1) / self.elements_num \
               + torch.norm(self.linear_map.bias, 1) / self.bias_num

    def get_L2_reg(self) -> torch.Tensor:
        return torch.norm(self.linear_map.weight, 2).pow(2) / self.elements_num \
               + torch.norm(self.linear_map.bias, 2).pow(2) / self.bias_num

    def _init_weight(self):
        torch.nn.init.xavier_uniform_(self.linear_map.weight)


class LinearLogSoftMaxEnvClassifier(EnvClassifier):
    def __init__(self, factor_dim, env_num):
        super(LinearLogSoftMaxEnvClassifier, self).__init__()
        self.linear_map: nn.Linear = nn.Linear(factor_dim, env_num)
        self.classifier_func = nn.LogSoftmax(dim=1)
        self._init_weight()
        self.elements_num: float = float(factor_dim * env_num)
        self.bias_num: float = float(env_num)

    def forward(self, invariant_preferences):
        result: torch.Tensor = self.linear_map(invariant_preferences)
        result = self.classifier_func(result)
        return result

    def get_L1_reg(self) -> torch.Tensor:
        return torch.norm(self.linear_map.weight, 1) / self.elements_num \
               + torch.norm(self.linear_map.bias, 1) / self.bias_num

    def get_L2_reg(self) -> torch.Tensor:
        return torch.norm(self.linear_map.weight, 2).pow(2) / self.elements_num \
               + torch.norm(self.linear_map.bias, 2).pow(2) / self.bias_num

    def _init_weight(self):
        torch.nn.init.xavier_uniform_(self.linear_map.weight)


class LinearImplicitScorePredictor(ScorePredictor):
    def __init__(self, factor_dim):
        super(LinearImplicitScorePredictor, self).__init__()
        self.linear_map: nn.Linear = nn.Linear(factor_dim, 1)
        # Sigmoid is necessary for BCE loss
        self.output_func = nn.Sigmoid()
        self._init_weight()
        self.elements_num: float = float(factor_dim)

    def forward(self, invariant_preferences):
        result: torch.Tensor = self.linear_map(invariant_preferences)
        result = self.output_func(result)
        return result

    def get_L1_reg(self) -> torch.Tensor:
        return torch.norm(self.linear_map.weight, 1) / self.elements_num \
               + torch.norm(self.linear_map.bias, 1)

    def get_L2_reg(self) -> torch.Tensor:
        return torch.norm(self.linear_map.weight, 2).pow(2) / self.elements_num \
               + torch.norm(self.linear_map.bias, 2).pow(2)

    def _init_weight(self):
        torch.nn.init.xavier_uniform_(self.linear_map.weight)


class LinearExplicitScorePredictor(ScorePredictor):
    def __init__(self, factor_dim):
        super(LinearExplicitScorePredictor, self).__init__()
        self.linear_map: nn.Linear = nn.Linear(factor_dim, 1)
        self._init_weight()
        self.elements_num: float = float(factor_dim)

    def forward(self, invariant_preferences):
        result: torch.Tensor = self.linear_map(invariant_preferences)
        return result

    def get_L1_reg(self) -> torch.Tensor:
        return torch.norm(self.linear_map.weight, 1) / self.elements_num \
               + torch.norm(self.linear_map.bias, 1)

    def get_L2_reg(self) -> torch.Tensor:
        return torch.norm(self.linear_map.weight, 2).pow(2) / self.elements_num \
               + torch.norm(self.linear_map.bias, 2).pow(2)

    def _init_weight(self):
        torch.nn.init.xavier_uniform_(self.linear_map.weight)


class InvPrefImplicit(GeneralDebiasImplicitRecommender):
    def __init__(
            self, user_num: int, item_num: int, env_num: int, factor_num: int, reg_only_embed: bool = False,
            reg_env_embed: bool = True
    ):
        super(InvPrefImplicit, self).__init__(
            user_num=user_num, item_num=item_num, env_num=env_num
        )

        self.factor_num: int = factor_num

        self.embed_user_invariant = nn.Embedding(user_num, factor_num)
        self.embed_item_invariant = nn.Embedding(item_num, factor_num)

        self.embed_user_env_aware = nn.Embedding(user_num, factor_num)
        self.embed_item_env_aware = nn.Embedding(item_num, factor_num)

        self.embed_env = nn.Embedding(env_num, factor_num)

        self.env_classifier: EnvClassifier = LinearLogSoftMaxEnvClassifier(factor_num, env_num)
        self.output_func = nn.Sigmoid()

        self.reg_only_embed: bool = reg_only_embed

        self.reg_env_embed: bool = reg_env_embed

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.embed_user_invariant.weight, std=0.01)
        nn.init.normal_(self.embed_item_invariant.weight, std=0.01)
        nn.init.normal_(self.embed_user_env_aware.weight, std=0.01)
        nn.init.normal_(self.embed_item_env_aware.weight, std=0.01)
        nn.init.normal_(self.embed_env.weight, std=0.01)

    def forward(self, users_id, items_id, envs_id, alpha):
        users_embed_invariant: torch.Tensor = self.embed_user_invariant(users_id)
        items_embed_invariant: torch.Tensor = self.embed_item_invariant(items_id)

        users_embed_env_aware: torch.Tensor = self.embed_user_env_aware(users_id)
        items_embed_env_aware: torch.Tensor = self.embed_item_env_aware(items_id)

        envs_embed: torch.Tensor = self.embed_env(envs_id)

        invariant_preferences: torch.Tensor = users_embed_invariant * items_embed_invariant
        env_aware_preferences: torch.Tensor = users_embed_env_aware * items_embed_env_aware * envs_embed

        invariant_score: torch.Tensor = self.output_func(torch.sum(invariant_preferences, dim=1))
        env_aware_mid_score: torch.Tensor = self.output_func(torch.sum(env_aware_preferences, dim=1))
        env_aware_score: torch.Tensor = invariant_score * env_aware_mid_score

        reverse_invariant_preferences: torch.Tensor = ReverseLayerF.apply(invariant_preferences, alpha)
        env_outputs: torch.Tensor = self.env_classifier(reverse_invariant_preferences)

        return invariant_score.reshape(-1), env_aware_score.reshape(-1), env_outputs.reshape(-1, self.env_num)

    def get_users_reg(self, users_id, norm: int):
        invariant_embed_gmf: torch.Tensor = self.embed_user_invariant(users_id)
        env_aware_embed_gmf: torch.Tensor = self.embed_user_env_aware(users_id)
        if norm == 2:
            reg_loss: torch.Tensor = \
                (env_aware_embed_gmf.norm(2).pow(2) + invariant_embed_gmf.norm(2).pow(2)) \
                / (float(len(users_id)) * float(self.factor_num) * 2)
        elif norm == 1:
            reg_loss: torch.Tensor = \
                (env_aware_embed_gmf.norm(1) + invariant_embed_gmf.norm(1)) \
                / (float(len(users_id)) * float(self.factor_num) * 2)
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_items_reg(self, items_id, norm: int):
        invariant_embed_gmf: torch.Tensor = self.embed_item_invariant(items_id)
        env_aware_embed_gmf: torch.Tensor = self.embed_item_env_aware(items_id)
        if norm == 2:
            reg_loss: torch.Tensor = \
                (env_aware_embed_gmf.norm(2).pow(2) + invariant_embed_gmf.norm(2).pow(2)) \
                / (float(len(items_id)) * float(self.factor_num) * 2)
        elif norm == 1:
            reg_loss: torch.Tensor = \
                (env_aware_embed_gmf.norm(1) + invariant_embed_gmf.norm(1)) \
                / (float(len(items_id)) * float(self.factor_num) * 2)
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_envs_reg(self, envs_id, norm: int):
        embed_gmf: torch.Tensor = self.embed_env(envs_id)
        if norm == 2:
            reg_loss: torch.Tensor = embed_gmf.norm(2).pow(2) / (float(len(envs_id)) * float(self.factor_num))
        elif norm == 1:
            reg_loss: torch.Tensor = embed_gmf.norm(1) / (float(len(envs_id)) * float(self.factor_num))
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_L2_reg(self, users_id, items_id, envs_id):
        if not self.reg_only_embed:
            result = self.env_classifier.get_L2_reg()
            result = result + (
                    self.get_users_reg(users_id, 2) + self.get_items_reg(items_id, 2))
        else:
            result = self.get_users_reg(users_id, 2) + self.get_items_reg(items_id, 2)

        if self.reg_env_embed:
            result = result + self.get_envs_reg(envs_id, 2)
        # print('L2', result)
        return result

    def get_L1_reg(self, users_id, items_id, envs_id):
        if not self.reg_only_embed:
            result = self.env_classifier.get_L1_reg()
            result = result + (
                    self.get_users_reg(users_id, 1) + self.get_items_reg(items_id, 1))
        else:
            result = self.get_users_reg(users_id, 1) + self.get_items_reg(items_id, 1)
        if self.reg_env_embed:
            result = result + self.get_envs_reg(envs_id, 1)
        # print('L2', result)
        return result

    def predict(self, users_id):
        users_embed_gmf: torch.Tensor = self.embed_user_invariant(users_id)
        items_embed_gmf: torch.Tensor = self.embed_item_invariant.weight

        user_to_cat = []
        for i in range(users_embed_gmf.shape[0]):
            tmp: torch.Tensor = users_embed_gmf[i:i + 1, :]
            tmp = tmp.repeat(items_embed_gmf.shape[0], 1)
            user_to_cat.append(tmp)
        users_emb_cat: torch.Tensor = torch.cat(user_to_cat, dim=0)
        items_emb_cat: torch.Tensor = items_embed_gmf.repeat(users_embed_gmf.shape[0], 1)

        invariant_preferences: torch.Tensor = users_emb_cat * items_emb_cat
        invariant_score: torch.Tensor = self.output_func(torch.sum(invariant_preferences, dim=1))
        return invariant_score.reshape(users_id.shape[0], items_embed_gmf.shape[0])

    def cluster_predict(self, users_id, items_id, envs_id) -> torch.Tensor:
        _, env_aware_score, _ = self.forward(users_id, items_id, envs_id, 0.)
        return env_aware_score


class InvPrefExplicit(GeneralDebiasExplicitRecommender):
    def __init__(
            self, user_num: int, item_num: int, env_num: int, factor_num: int, reg_only_embed: bool = False,
            reg_env_embed: bool = True
    ):
        super(InvPrefExplicit, self).__init__(
            user_num=user_num, item_num=item_num, env_num=env_num
        )

        self.factor_num: int = factor_num

        self.embed_user_invariant = nn.Embedding(user_num, factor_num)
        self.embed_item_invariant = nn.Embedding(item_num, factor_num)

        self.embed_user_env_aware = nn.Embedding(user_num, factor_num)
        self.embed_item_env_aware = nn.Embedding(item_num, factor_num)

        self.embed_env = nn.Embedding(env_num, factor_num)

        self.env_classifier: EnvClassifier = LinearLogSoftMaxEnvClassifier(factor_num, env_num)

        self.reg_only_embed: bool = reg_only_embed

        self.reg_env_embed: bool = reg_env_embed

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.embed_user_invariant.weight, std=0.01)
        nn.init.normal_(self.embed_item_invariant.weight, std=0.01)
        nn.init.normal_(self.embed_user_env_aware.weight, std=0.01)
        nn.init.normal_(self.embed_item_env_aware.weight, std=0.01)
        nn.init.normal_(self.embed_env.weight, std=0.01)

    def forward(self, users_id, items_id, envs_id, alpha):
        users_embed_invariant: torch.Tensor = self.embed_user_invariant(users_id)
        items_embed_invariant: torch.Tensor = self.embed_item_invariant(items_id)

        users_embed_env_aware: torch.Tensor = self.embed_user_env_aware(users_id)
        items_embed_env_aware: torch.Tensor = self.embed_item_env_aware(items_id)

        envs_embed: torch.Tensor = self.embed_env(envs_id)

        invariant_preferences: torch.Tensor = users_embed_invariant * items_embed_invariant
        env_aware_preferences: torch.Tensor = users_embed_env_aware * items_embed_env_aware * envs_embed

        invariant_score: torch.Tensor = torch.sum(invariant_preferences, dim=1)
        env_aware_mid_score: torch.Tensor = torch.sum(env_aware_preferences, dim=1)
        env_aware_score: torch.Tensor = invariant_score + env_aware_mid_score

        reverse_invariant_preferences: torch.Tensor = ReverseLayerF.apply(invariant_preferences, alpha)
        env_outputs: torch.Tensor = self.env_classifier(reverse_invariant_preferences)

        return invariant_score.reshape(-1), env_aware_score.reshape(-1), env_outputs.reshape(-1, self.env_num)

    def get_users_reg(self, users_id, norm: int):
        invariant_embed_gmf: torch.Tensor = self.embed_user_invariant(users_id)
        env_aware_embed_gmf: torch.Tensor = self.embed_user_env_aware(users_id)
        if norm == 2:
            reg_loss: torch.Tensor = \
                (env_aware_embed_gmf.norm(2).pow(2) + invariant_embed_gmf.norm(2).pow(2)) \
                / (float(len(users_id)) * float(self.factor_num) * 2)
        elif norm == 1:
            reg_loss: torch.Tensor = \
                (env_aware_embed_gmf.norm(1) + invariant_embed_gmf.norm(1)) \
                / (float(len(users_id)) * float(self.factor_num) * 2)
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_items_reg(self, items_id, norm: int):
        invariant_embed_gmf: torch.Tensor = self.embed_item_invariant(items_id)
        env_aware_embed_gmf: torch.Tensor = self.embed_item_env_aware(items_id)
        if norm == 2:
            reg_loss: torch.Tensor = \
                (env_aware_embed_gmf.norm(2).pow(2) + invariant_embed_gmf.norm(2).pow(2)) \
                / (float(len(items_id)) * float(self.factor_num) * 2)
        elif norm == 1:
            reg_loss: torch.Tensor = \
                (env_aware_embed_gmf.norm(1) + invariant_embed_gmf.norm(1)) \
                / (float(len(items_id)) * float(self.factor_num) * 2)
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_envs_reg(self, envs_id, norm: int):
        embed_gmf: torch.Tensor = self.embed_env(envs_id)
        if norm == 2:
            reg_loss: torch.Tensor = embed_gmf.norm(2).pow(2) / (float(len(envs_id)) * float(self.factor_num))
        elif norm == 1:
            reg_loss: torch.Tensor = embed_gmf.norm(1) / (float(len(envs_id)) * float(self.factor_num))
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_L2_reg(self, users_id, items_id, envs_id):
        if not self.reg_only_embed:
            result = self.env_classifier.get_L2_reg()
            result = result + (
                    self.get_users_reg(users_id, 2) + self.get_items_reg(items_id, 2))
        else:
            result = self.get_users_reg(users_id, 2) + self.get_items_reg(items_id, 2)

        if self.reg_env_embed:
            result = result + self.get_envs_reg(envs_id, 2)
        # print('L2', result)
        return result

    def get_L1_reg(self, users_id, items_id, envs_id):
        if not self.reg_only_embed:
            result = self.env_classifier.get_L1_reg()
            result = result + (
                    self.get_users_reg(users_id, 1) + self.get_items_reg(items_id, 1))
        else:
            result = self.get_users_reg(users_id, 1) + self.get_items_reg(items_id, 1)
        if self.reg_env_embed:
            result = result + self.get_envs_reg(envs_id, 1)
        # print('L2', result)
        return result

    def predict(self, users_id, items_id):
        users_embed_invariant: torch.Tensor = self.embed_user_invariant(users_id)
        items_embed_invariant: torch.Tensor = self.embed_item_invariant(items_id)
        invariant_preferences: torch.Tensor = users_embed_invariant * items_embed_invariant
        invariant_score: torch.Tensor = torch.sum(invariant_preferences, dim=1)
        return invariant_score.reshape(-1)

    def cluster_predict(self, users_id, items_id, envs_id) -> torch.Tensor:
        _, env_aware_score, _ = self.forward(users_id, items_id, envs_id, 0.)
        return env_aware_score


