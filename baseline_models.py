import copy
import math
import numpy as np

from torch import nn
import torch
from torch.autograd import Variable

from models import BasicRecommender, LinearImplicitScorePredictor, BasicExplicitRecommender


class PureMatrixFactorization(BasicRecommender):
    def __init__(self, user_num: int, item_num: int, factor_num: int):
        super(PureMatrixFactorization, self).__init__(user_num, item_num)
        self.factor_num: int = factor_num
        self.user_emb: nn.Embedding = nn.Embedding(user_num, factor_num)
        self.item_emb: nn.Embedding = nn.Embedding(item_num, factor_num)
        self.output_func = nn.Sigmoid()
        self.loss_func = nn.BCELoss()

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, users_id: torch.Tensor, items_id: torch.Tensor, ground_truth: torch.Tensor = None):
        users_emb: torch.Tensor = self.user_emb(users_id)
        items_emb: torch.Tensor = self.item_emb(items_id)

        ratings: torch.Tensor = torch.sum(users_emb * items_emb, dim=1)
        final_ratings: torch.Tensor = self.output_func(ratings)
        if ground_truth is not None:
            return self.loss_func(final_ratings, ground_truth)
        else:
            return final_ratings

    def get_users_reg(self, users_id, norm: int):
        embed_gmf: torch.Tensor = self.user_emb(users_id)
        if norm == 2:
            reg_loss: torch.Tensor = embed_gmf.norm(2).pow(2) / (float(len(users_id)) * float(self.factor_num))
        elif norm == 1:
            reg_loss: torch.Tensor = embed_gmf.norm(1) / (float(len(users_id)) * float(self.factor_num))
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_items_reg(self, items_id, norm: int):
        embed_gmf: torch.Tensor = self.item_emb(items_id)
        if norm == 2:
            reg_loss: torch.Tensor = embed_gmf.norm(2).pow(2) / (float(len(items_id)) * float(self.factor_num))
        elif norm == 1:
            reg_loss: torch.Tensor = embed_gmf.norm(1) / (float(len(items_id)) * float(self.factor_num))
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_L1_reg(self, users_id, items_id) -> torch.Tensor:
        return self.get_users_reg(users_id, 1) + self.get_items_reg(items_id, 1)

    def get_L2_reg(self, users_id, items_id) -> torch.Tensor:
        return self.get_users_reg(users_id, 2) + self.get_items_reg(items_id, 2)

    def predict(self, users_id) -> torch.Tensor:
        users_emb: torch.Tensor = self.user_emb(users_id)
        items_emb: torch.Tensor = self.item_emb.weight

        ratings: torch.Tensor = torch.matmul(users_emb, items_emb.t())
        return self.output_func(ratings)


class LinearTransMatrixFactorization(BasicRecommender):
    def __init__(self, user_num: int, item_num: int, factor_num: int):
        super(LinearTransMatrixFactorization, self).__init__(user_num, item_num)
        self.factor_num: int = factor_num
        self.user_emb: nn.Embedding = nn.Embedding(user_num, factor_num)
        self.item_emb: nn.Embedding = nn.Embedding(item_num, factor_num)
        self.linear_predictor: LinearImplicitScorePredictor = LinearImplicitScorePredictor(factor_num)
        self.loss_func = nn.BCELoss()

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, users_id: torch.Tensor, items_id: torch.Tensor, ground_truth: torch.Tensor):
        users_emb: torch.Tensor = self.user_emb(users_id)
        items_emb: torch.Tensor = self.item_emb(items_id)

        preferences: torch.Tensor = users_emb * items_emb
        final_ratings: torch.Tensor = self.linear_predictor(preferences).reshape(-1)
        return self.loss_func(final_ratings, ground_truth)

    def get_users_reg(self, users_id, norm: int):
        embed_gmf: torch.Tensor = self.user_emb(users_id)
        if norm == 2:
            reg_loss: torch.Tensor = embed_gmf.norm(2).pow(2) / (float(len(users_id)) * float(self.factor_num))
        elif norm == 1:
            reg_loss: torch.Tensor = embed_gmf.norm(1) / (float(len(users_id)) * float(self.factor_num))
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_items_reg(self, items_id, norm: int):
        embed_gmf: torch.Tensor = self.item_emb(items_id)
        if norm == 2:
            reg_loss: torch.Tensor = embed_gmf.norm(2).pow(2) / (float(len(items_id)) * float(self.factor_num))
        elif norm == 1:
            reg_loss: torch.Tensor = embed_gmf.norm(1) / (float(len(items_id)) * float(self.factor_num))
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_L1_reg(self, users_id, items_id) -> torch.Tensor:
        return self.get_users_reg(users_id, 1) + self.get_items_reg(items_id, 1) + self.linear_predictor.get_L1_reg()

    def get_L2_reg(self, users_id, items_id) -> torch.Tensor:
        return self.get_users_reg(users_id, 2) + self.get_items_reg(items_id, 2) + self.linear_predictor.get_L2_reg()

    def predict(self, users_id) -> torch.Tensor:
        users_embed_gmf: torch.Tensor = self.user_emb(users_id)
        items_embed_gmf: torch.Tensor = self.item_emb.weight

        user_to_cat = []
        for i in range(users_embed_gmf.shape[0]):
            tmp: torch.Tensor = users_embed_gmf[i:i + 1, :]
            tmp = tmp.repeat(items_embed_gmf.shape[0], 1)
            user_to_cat.append(tmp)
        users_emb_cat: torch.Tensor = torch.cat(user_to_cat, dim=0)
        items_emb_cat: torch.Tensor = items_embed_gmf.repeat(users_embed_gmf.shape[0], 1)

        preferences: torch.Tensor = users_emb_cat * items_emb_cat
        final_ratings: torch.Tensor = self.linear_predictor(preferences)

        return final_ratings.reshape(users_embed_gmf.shape[0], items_embed_gmf.shape[0])


class MACRMatrixFactorization(BasicRecommender):
    def __init__(self, user_num: int, item_num: int, factor_num: int,
                 const_c: float, item_coe: float, user_coe: float
                 ):
        super(MACRMatrixFactorization, self).__init__(user_num, item_num)
        self.factor_num: int = factor_num
        self.user_emb: nn.Embedding = nn.Embedding(user_num, factor_num)
        self.item_emb: nn.Embedding = nn.Embedding(item_num, factor_num)
        self.output_func = nn.Sigmoid()
        self.const_c: float = const_c

        self.user_predictor: LinearImplicitScorePredictor = LinearImplicitScorePredictor(factor_num)
        self.item_predictor: LinearImplicitScorePredictor = LinearImplicitScorePredictor(factor_num)

        self.loss_func = nn.BCELoss()

        self.item_coe: float = item_coe
        self.user_coe: float = user_coe

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, users_id: torch.Tensor, items_id: torch.Tensor, ground_truth: torch.Tensor):
        users_emb: torch.Tensor = self.user_emb(users_id)
        items_emb: torch.Tensor = self.item_emb(items_id)

        ratings: torch.Tensor = torch.sum(users_emb * items_emb, dim=1)
        interaction_ratings: torch.Tensor = self.output_func(ratings)

        user_ratings: torch.Tensor = self.user_predictor(users_emb).reshape(-1)
        item_ratings: torch.Tensor = self.item_predictor(items_emb).reshape(-1)

        final_ratings: torch.Tensor = interaction_ratings * user_ratings * item_ratings

        # print(interaction_ratings.shape, user_ratings.shape, item_ratings.shape, final_ratings.shape)

        loss: torch.Tensor = self.loss_func(final_ratings, ground_truth) \
                             + self.loss_func(user_ratings, ground_truth) * self.user_coe \
                             + self.loss_func(item_ratings, ground_truth) * self.item_coe

        return loss

    def get_users_reg(self, users_id, norm: int):
        embed_gmf: torch.Tensor = self.user_emb(users_id)
        if norm == 2:
            reg_loss: torch.Tensor = embed_gmf.norm(2).pow(2) / (float(len(users_id)) * float(self.factor_num))
        elif norm == 1:
            reg_loss: torch.Tensor = embed_gmf.norm(1) / (float(len(users_id)) * float(self.factor_num))
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_items_reg(self, items_id, norm: int):
        embed_gmf: torch.Tensor = self.item_emb(items_id)
        if norm == 2:
            reg_loss: torch.Tensor = embed_gmf.norm(2).pow(2) / (float(len(items_id)) * float(self.factor_num))
        elif norm == 1:
            reg_loss: torch.Tensor = embed_gmf.norm(1) / (float(len(items_id)) * float(self.factor_num))
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_L1_reg(self, users_id, items_id) -> torch.Tensor:
        return self.get_users_reg(users_id, 1) + self.get_items_reg(items_id, 1)

    def get_L2_reg(self, users_id, items_id) -> torch.Tensor:
        return self.get_users_reg(users_id, 2) + self.get_items_reg(items_id, 2)

    def predict(self, users_id) -> torch.Tensor:
        users_embed_gmf: torch.Tensor = self.user_emb(users_id)
        items_embed_gmf: torch.Tensor = self.item_emb.weight

        user_to_cat = []
        for i in range(users_embed_gmf.shape[0]):
            tmp: torch.Tensor = users_embed_gmf[i:i + 1, :]
            tmp = tmp.repeat(items_embed_gmf.shape[0], 1)
            user_to_cat.append(tmp)
        users_emb_cat: torch.Tensor = torch.cat(user_to_cat, dim=0)
        items_emb_cat: torch.Tensor = items_embed_gmf.repeat(users_embed_gmf.shape[0], 1)

        ratings: torch.Tensor = torch.sum(users_emb_cat * items_emb_cat, dim=1)
        interaction_ratings: torch.Tensor = self.output_func(ratings)

        user_ratings: torch.Tensor = self.user_predictor(users_emb_cat)
        item_ratings: torch.Tensor = self.item_predictor(items_emb_cat)

        temp = (interaction_ratings - self.const_c).reshape(user_ratings.shape[0], -1)

        # print(interaction_ratings.shape, user_ratings.shape, item_ratings.shape, temp.shape)

        final_ratings: torch.Tensor = temp * user_ratings * item_ratings

        return final_ratings.reshape(users_id.shape[0], items_embed_gmf.shape[0])


class ExposureMatrixFactorization(PureMatrixFactorization):
    def __init__(
            self, user_num: int, item_num: int, factor_num: int,
    ):
        super(ExposureMatrixFactorization, self).__init__(user_num, item_num, factor_num)
        self.loss_func = nn.BCELoss(reduction='none')

    def forward(self, users_id: torch.Tensor, items_id: torch.Tensor, ground_truth: torch.Tensor):
        users_emb: torch.Tensor = self.user_emb(users_id)
        items_emb: torch.Tensor = self.item_emb(items_id)

        ratings: torch.Tensor = torch.sum(users_emb * items_emb, dim=1)
        final_ratings: torch.Tensor = self.output_func(ratings)
        return self.loss_func(final_ratings, ground_truth)

    def calculate_exposure_probability(self, user_id: torch.Tensor, lam_y: float, mu: torch.Tensor, eps: float):
        p_ex: torch.Tensor = math.sqrt(lam_y / 2 * float(np.pi)) * torch.exp(-lam_y * self.predict(user_id) ** 2 / 2)
        probability: torch.Tensor = (p_ex + eps) / (p_ex + eps + (1 - mu) / mu)
        # print(probability.shape)
        return probability.detach()


class OneLinear(nn.Module):
    def __init__(self, n):
        super().__init__()

        self.data_bias = nn.Embedding(n, 1)
        self.init_embedding()

    def init_embedding(self):
        self.data_bias.weight.data *= 0.001

    def forward(self, values):
        d_bias = self.data_bias(values)
        return d_bias.squeeze()


class TwoLinear(nn.Module):
    """
    linear model: u + i + r / o
    """

    def __init__(self, n_user, n_item):
        super().__init__()

        self.user_bias = nn.Embedding(n_user, 1)
        self.item_bias = nn.Embedding(n_item, 1)
        self.init_embedding(0)

    def init_embedding(self, init):
        nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a=init)
        nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a=init)

    def forward(self, users, items):
        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)
        preds = u_bias + i_bias
        return preds.squeeze()


class ThreeLinear(nn.Module):
    """
    linear model: u + i + r / o
    """

    def __init__(self, n_user, n_item, n):
        super().__init__()

        self.user_bias = nn.Embedding(n_user, 1)
        self.item_bias = nn.Embedding(n_item, 1)
        self.data_bias = nn.Embedding(n, 1)
        self.init_embedding(0)

    def init_embedding(self, init):
        nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a=init)
        nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a=init)
        nn.init.kaiming_normal_(self.data_bias.weight, mode='fan_out', a=init)
        self.data_bias.weight.data *= 0.001

    def forward(self, users, items, values):
        u_bias = self.user_bias(users)
        i_bias = self.item_bias(items)
        d_bias = self.data_bias(values)

        preds = u_bias + i_bias + d_bias
        return preds.squeeze()


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaEmbed(MetaModule):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        ignore = nn.Embedding(dim_1, dim_2)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', None)

    def forward(self):
        return self.weight

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaMF(MetaModule, BasicRecommender):
    """
    Base module for matrix factorization.
    """

    def get_users_reg(self, users_id, norm: int):
        embed_gmf: torch.Tensor = self.user_emb.weight[users_id]
        if norm == 2:
            reg_loss: torch.Tensor = embed_gmf.norm(2).pow(2) / (float(len(users_id)) * float(self.dim))
        elif norm == 1:
            reg_loss: torch.Tensor = embed_gmf.norm(1) / (float(len(users_id)) * float(self.dim))
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_items_reg(self, items_id, norm: int):
        embed_gmf: torch.Tensor = self.item_emb.weight[items_id]
        if norm == 2:
            reg_loss: torch.Tensor = embed_gmf.norm(2).pow(2) / (float(len(items_id)) * float(self.dim))
        elif norm == 1:
            reg_loss: torch.Tensor = embed_gmf.norm(1) / (float(len(items_id)) * float(self.dim))
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_L1_reg(self, users_id, items_id) -> torch.Tensor:
        return self.get_users_reg(users_id, 1) + self.get_items_reg(items_id, 1)

    def get_L2_reg(self, users_id, items_id) -> torch.Tensor:
        return self.get_users_reg(users_id, 2) + self.get_items_reg(items_id, 2)

    def predict(self, users_id) -> torch.Tensor:
        users_emb: torch.Tensor = self.user_emb.weight[users_id]
        items_emb: torch.Tensor = self.item_emb.weight

        ratings: torch.Tensor = torch.matmul(users_emb, items_emb.t())
        return self.output_func(ratings)

    def __init__(self, n_user, n_item, dim: int, init=None):
        super().__init__(user_num=n_user, item_num=n_item)

        self.user_emb = MetaEmbed(n_user, dim)
        self.item_emb = MetaEmbed(n_item, dim)
        self.dim = dim

        self.output_func: nn.Sigmoid = nn.Sigmoid()
        if init is not None:
            self.init_embedding(init)
        else:
            self.init_embedding(0)

    def init_embedding(self, init):

        nn.init.kaiming_normal_(self.user_emb.weight, mode='fan_out', a=init)
        nn.init.kaiming_normal_(self.item_emb.weight, mode='fan_out', a=init)

    def forward(self, users, items):
        u_latent = self.user_emb.weight[users]
        i_latent = self.item_emb.weight[items]

        preds = torch.sum(u_latent * i_latent, dim=1, keepdim=True)
        return self.output_func(preds.reshape(-1))

    def parameters(self, recurse: bool = True):
        return self.params()


class MetaMFExplicit(MetaModule, BasicExplicitRecommender):
    """
    Base module for matrix factorization.
    """

    def get_users_reg(self, users_id, norm: int):
        embed_gmf: torch.Tensor = self.user_emb.weight[users_id]
        if norm == 2:
            reg_loss: torch.Tensor = embed_gmf.norm(2).pow(2) / (float(len(users_id)) * float(self.dim))
        elif norm == 1:
            reg_loss: torch.Tensor = embed_gmf.norm(1) / (float(len(users_id)) * float(self.dim))
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_items_reg(self, items_id, norm: int):
        embed_gmf: torch.Tensor = self.item_emb.weight[items_id]
        if norm == 2:
            reg_loss: torch.Tensor = embed_gmf.norm(2).pow(2) / (float(len(items_id)) * float(self.dim))
        elif norm == 1:
            reg_loss: torch.Tensor = embed_gmf.norm(1) / (float(len(items_id)) * float(self.dim))
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_L1_reg(self, users_id, items_id) -> torch.Tensor:
        return self.get_users_reg(users_id, 1) + self.get_items_reg(items_id, 1)

    def get_L2_reg(self, users_id, items_id) -> torch.Tensor:
        return self.get_users_reg(users_id, 2) + self.get_items_reg(items_id, 2)

    def predict(self, users_id, items_id) -> torch.Tensor:
        return self.forward(users_id, items_id)

    def __init__(self, n_user, n_item, dim: int, init=None):
        super().__init__(user_num=n_user, item_num=n_item)

        self.user_emb = MetaEmbed(n_user, dim)
        self.item_emb = MetaEmbed(n_item, dim)
        self.dim = dim

        if init is not None:
            self.init_embedding(init)
        else:
            self.init_embedding(0)

    def init_embedding(self, init):

        nn.init.kaiming_normal_(self.user_emb.weight, mode='fan_out', a=init)
        nn.init.kaiming_normal_(self.item_emb.weight, mode='fan_out', a=init)

    def forward(self, users, items):
        u_latent = self.user_emb.weight[users]
        i_latent = self.item_emb.weight[items]

        preds = torch.sum(u_latent * i_latent, dim=1, keepdim=True)
        return preds.reshape(-1)

    def parameters(self, recurse: bool = True):
        return self.params()


class CausEMatrixFactorization(BasicRecommender):
    def __init__(self, user_num: int, item_num: int, factor_num: int):
        super(CausEMatrixFactorization, self).__init__(user_num, item_num)
        self.factor_num: int = factor_num
        self.user_emb: nn.Embedding = nn.Embedding(user_num, factor_num)
        self.item_emb: nn.Embedding = nn.Embedding(item_num, factor_num)
        self.teacher_user_emb: nn.Embedding = nn.Embedding(user_num, factor_num)
        self.teacher_item_emb: nn.Embedding = nn.Embedding(item_num, factor_num)
        self.output_func = nn.Sigmoid()
        self.loss_func = nn.BCELoss()

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.normal_(self.teacher_user_emb.weight, std=0.01)
        nn.init.normal_(self.teacher_item_emb.weight, std=0.01)

    def forward(
            self,
            users_id: torch.Tensor,
            items_id: torch.Tensor,
            train_teacher: bool,
            ground_truth: torch.Tensor = None
    ):
        if not train_teacher:
            users_emb: torch.Tensor = self.user_emb(users_id)
            items_emb: torch.Tensor = self.item_emb(items_id)
        else:
            users_emb: torch.Tensor = self.teacher_user_emb(users_id)
            items_emb: torch.Tensor = self.teacher_item_emb(items_id)

        ratings: torch.Tensor = torch.sum(users_emb * items_emb, dim=1)
        final_ratings: torch.Tensor = self.output_func(ratings)
        if ground_truth is not None:
            return self.loss_func(final_ratings, ground_truth)
        else:
            return final_ratings

    def get_users_reg(self, users_id, norm: int, train_teacher: bool):
        if not train_teacher:
            embed_gmf: torch.Tensor = self.user_emb(users_id)
        else:
            embed_gmf: torch.Tensor = self.teacher_user_emb(users_id)
        if norm == 2:
            reg_loss: torch.Tensor = embed_gmf.norm(2).pow(2) / (float(len(users_id)) * float(self.factor_num))
        elif norm == 1:
            reg_loss: torch.Tensor = embed_gmf.norm(1) / (float(len(users_id)) * float(self.factor_num))
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_items_reg(self, items_id, norm: int, train_teacher: bool):
        if not train_teacher:
            embed_gmf: torch.Tensor = self.user_emb(items_id)
        else:
            embed_gmf: torch.Tensor = self.teacher_user_emb(items_id)
        if norm == 2:
            reg_loss: torch.Tensor = embed_gmf.norm(2).pow(2) / (float(len(items_id)) * float(self.factor_num))
        elif norm == 1:
            reg_loss: torch.Tensor = embed_gmf.norm(1) / (float(len(items_id)) * float(self.factor_num))
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_L1_reg(self, users_id, items_id, train_teacher) -> torch.Tensor:
        return self.get_users_reg(users_id, 1, train_teacher) + self.get_items_reg(items_id, 1, train_teacher)

    def get_L2_reg(self, users_id, items_id, train_teacher) -> torch.Tensor:
        return self.get_users_reg(users_id, 2, train_teacher) + self.get_items_reg(items_id, 2, train_teacher)

    def predict(self, users_id) -> torch.Tensor:
        users_emb: torch.Tensor = self.user_emb(users_id)
        items_emb: torch.Tensor = self.item_emb.weight

        ratings: torch.Tensor = torch.matmul(users_emb, items_emb.t())
        return self.output_func(ratings)

    def item_teacher_reg(self, items_id):
        student_emb: torch.Tensor = self.item_emb(items_id)
        teacher_emb: torch.Tensor = self.teacher_item_emb(items_id).detach()

        distance_matrix: torch.Tensor = (student_emb - teacher_emb) ** 2
        # distance_matrix: torch.Tensor = torch.abs(student_emb - teacher_emb)

        return torch.mean(distance_matrix)

    def user_teacher_reg(self, users_id):
        student_emb: torch.Tensor = self.user_emb(users_id)
        teacher_emb: torch.Tensor = self.teacher_user_emb(users_id).detach()

        distance_matrix: torch.Tensor = (student_emb - teacher_emb) ** 2

        return torch.mean(distance_matrix)


class PureExplicitMatrixFactorization(BasicExplicitRecommender):
    def __init__(self, user_num: int, item_num: int, factor_num: int):
        super(PureExplicitMatrixFactorization, self).__init__(user_num, item_num)
        self.factor_num: int = factor_num
        self.user_emb: nn.Embedding = nn.Embedding(user_num, factor_num)
        self.item_emb: nn.Embedding = nn.Embedding(item_num, factor_num)
        self.loss_func = nn.MSELoss()

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, users_id: torch.Tensor, items_id: torch.Tensor, ground_truth: torch.Tensor = None):
        users_emb: torch.Tensor = self.user_emb(users_id)
        items_emb: torch.Tensor = self.item_emb(items_id)

        ratings: torch.Tensor = torch.sum(users_emb * items_emb, dim=1)
        if ground_truth is not None:
            return self.loss_func(ratings, ground_truth)
        else:
            return ratings

    def get_users_reg(self, users_id, norm: int):
        embed_gmf: torch.Tensor = self.user_emb(users_id)
        if norm == 2:
            reg_loss: torch.Tensor = embed_gmf.norm(2).pow(2) / (float(len(users_id)) * float(self.factor_num))
        elif norm == 1:
            reg_loss: torch.Tensor = embed_gmf.norm(1) / (float(len(users_id)) * float(self.factor_num))
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_items_reg(self, items_id, norm: int):
        embed_gmf: torch.Tensor = self.item_emb(items_id)
        if norm == 2:
            reg_loss: torch.Tensor = embed_gmf.norm(2).pow(2) / (float(len(items_id)) * float(self.factor_num))
        elif norm == 1:
            reg_loss: torch.Tensor = embed_gmf.norm(1) / (float(len(items_id)) * float(self.factor_num))
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_L1_reg(self, users_id, items_id) -> torch.Tensor:
        return self.get_users_reg(users_id, 1) + self.get_items_reg(items_id, 1)

    def get_L2_reg(self, users_id, items_id) -> torch.Tensor:
        return self.get_users_reg(users_id, 2) + self.get_items_reg(items_id, 2)

    def predict(self, users_id, items_id) -> torch.Tensor:
        return self.forward(users_id, items_id)


class CausEExplicitMatrixFactorization(BasicExplicitRecommender):
    def __init__(self, user_num: int, item_num: int, factor_num: int):
        super(CausEExplicitMatrixFactorization, self).__init__(user_num, item_num)
        self.factor_num: int = factor_num
        self.user_emb: nn.Embedding = nn.Embedding(user_num, factor_num)
        self.item_emb: nn.Embedding = nn.Embedding(item_num, factor_num)
        self.teacher_user_emb: nn.Embedding = nn.Embedding(user_num, factor_num)
        self.teacher_item_emb: nn.Embedding = nn.Embedding(item_num, factor_num)
        self.loss_func = nn.MSELoss()

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.normal_(self.teacher_user_emb.weight, std=0.01)
        nn.init.normal_(self.teacher_item_emb.weight, std=0.01)

    def forward(
            self,
            users_id: torch.Tensor,
            items_id: torch.Tensor,
            train_teacher: bool,
            ground_truth: torch.Tensor = None
    ):
        if not train_teacher:
            users_emb: torch.Tensor = self.user_emb(users_id)
            items_emb: torch.Tensor = self.item_emb(items_id)
        else:
            users_emb: torch.Tensor = self.teacher_user_emb(users_id)
            items_emb: torch.Tensor = self.teacher_item_emb(items_id)

        ratings: torch.Tensor = torch.sum(users_emb * items_emb, dim=1).reshape(-1)
        if ground_truth is not None:
            return self.loss_func(ratings, ground_truth)
        else:
            return ratings

    def get_users_reg(self, users_id, norm: int, train_teacher: bool):
        if not train_teacher:
            embed_gmf: torch.Tensor = self.user_emb(users_id)
        else:
            embed_gmf: torch.Tensor = self.teacher_user_emb(users_id)
        if norm == 2:
            reg_loss: torch.Tensor = embed_gmf.norm(2).pow(2) / (float(len(users_id)) * float(self.factor_num))
        elif norm == 1:
            reg_loss: torch.Tensor = embed_gmf.norm(1) / (float(len(users_id)) * float(self.factor_num))
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_items_reg(self, items_id, norm: int, train_teacher: bool):
        if not train_teacher:
            embed_gmf: torch.Tensor = self.item_emb(items_id)
        else:
            embed_gmf: torch.Tensor = self.teacher_item_emb(items_id)
        if norm == 2:
            reg_loss: torch.Tensor = embed_gmf.norm(2).pow(2) / (float(len(items_id)) * float(self.factor_num))
        elif norm == 1:
            reg_loss: torch.Tensor = embed_gmf.norm(1) / (float(len(items_id)) * float(self.factor_num))
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_L1_reg(self, users_id, items_id, train_teacher) -> torch.Tensor:
        return self.get_users_reg(users_id, 1, train_teacher) + self.get_items_reg(items_id, 1, train_teacher)

    def get_L2_reg(self, users_id, items_id, train_teacher) -> torch.Tensor:
        return self.get_users_reg(users_id, 2, train_teacher) + self.get_items_reg(items_id, 2, train_teacher)

    def predict(self, users_id, items_id) -> torch.Tensor:
        return self.forward(users_id, items_id, train_teacher=False)

    def item_teacher_reg(self, items_id):
        student_emb: torch.Tensor = self.item_emb(items_id)
        teacher_emb: torch.Tensor = self.teacher_item_emb(items_id).detach()

        distance_matrix: torch.Tensor = (student_emb - teacher_emb) ** 2
        # distance_matrix: torch.Tensor = torch.abs(student_emb - teacher_emb)

        return torch.mean(distance_matrix)

    def user_teacher_reg(self, users_id):
        student_emb: torch.Tensor = self.user_emb(users_id)
        teacher_emb: torch.Tensor = self.teacher_user_emb(users_id).detach()

        distance_matrix: torch.Tensor = (student_emb - teacher_emb) ** 2

        return torch.mean(distance_matrix)
