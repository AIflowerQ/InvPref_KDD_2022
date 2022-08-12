import torch
import time
from baseline_models import ExposureMatrixFactorization, TwoLinear, ThreeLinear, OneLinear, CausEMatrixFactorization, \
    PureMatrixFactorization, CausEExplicitMatrixFactorization
from evaluate import ImplicitTestManager, ExplicitTestManager
from models import BasicRecommender, BasicExplicitRecommender
from train import BasicImplicitTrainManager, BasicUniformImplicitTrainManager, BasicUniformExplicitTrainManager, \
    BasicExplicitTrainManager
import numpy as np
from collections import Counter
from torch import nn

from utils import merge_dict, _mean_merge_dict_func, mini_batch, transfer_loss_dict_to_line_str


class ExpoMFTrainManager(BasicImplicitTrainManager):
    def __init__(
            self, model: ExposureMatrixFactorization, evaluator: ImplicitTestManager,
            device: torch.device,
            training_data: torch.Tensor, batch_size: int,
            epochs: int, evaluate_interval: int, lr: float,
            L2_coe: float, L1_coe: float, test_begin_epoch: int = 0,
            lam_y: float = 1.0, init_mu: float = 1e-2, a: float = 1.0, b: float = 1.0,
            expo_weight_exp: float = 1.0, eps: float = 1e-8, upd_expo_interval: int = 10
    ):
        super(ExpoMFTrainManager, self).__init__(
            model=model, evaluator=evaluator, device=device, training_data=training_data,
            batch_size=batch_size, epochs=epochs, evaluate_interval=evaluate_interval, lr=lr,
            L2_coe=L2_coe, L1_coe=L1_coe, test_begin_epoch=test_begin_epoch
        )
        self.model: ExposureMatrixFactorization = model
        self.lam_y: float = lam_y
        self.a: float = a
        self.b: float = b
        self.mu: torch.Tensor = torch.Tensor(init_mu * np.ones(model.item_num, dtype=float)).to(self.device)
        self.expo_weight_exp: float = expo_weight_exp
        self.eps: float = eps

        self.user_id_tensor: torch.LongTensor = torch.LongTensor(list(range(self.model.user_num))).to(self.device)
        self.exposure_probability: np.array = np.zeros([self.model.user_num, self.model.item_num])
        self.upd_expo_interval: int = upd_expo_interval

    def calculate_exposure_probability(self):
        self.model.eval()
        upd_batch_size: int = self.evaluator.batch_size
        probability_batch_list: list = []
        for batch_idx, batch_user in enumerate(mini_batch(upd_batch_size, self.user_id_tensor)):
            probability_batch: np.array = self.model.calculate_exposure_probability(
                user_id=batch_user,
                lam_y=self.lam_y,
                mu=self.mu,
                eps=self.eps
            ).cpu().numpy()
            probability_batch_list.append(probability_batch)
        self.exposure_probability = np.concatenate(probability_batch_list, axis=0)

        pos_index: torch.Tensor = torch.nonzero(self.scores_tensor).reshape(-1)

        self.exposure_probability[
            self.users_tensor[pos_index].cpu().detach(), self.items_tensor[pos_index].cpu().detach()
        ] = 1.0

    def upd_mu(self):
        self.model.eval()
        upd_batch_size: int = self.evaluator.batch_size
        probability_list: list = []
        for batch_idx, batch_user in enumerate(mini_batch(upd_batch_size, self.user_id_tensor)):
            probability_batch: torch.Tensor = self.model.calculate_exposure_probability(
                user_id=batch_user,
                lam_y=self.lam_y,
                mu=self.mu,
                eps=self.eps
            )
            probability_batch_sum: torch.Tensor = torch.sum(probability_batch, dim=0).reshape(1, -1)
            probability_list.append(probability_batch_sum)

        batch_sum_list_tensor: torch.Tensor = torch.cat(probability_list, dim=0)
        pro_sum: torch.Tensor = torch.sum(batch_sum_list_tensor, dim=0).reshape(-1)
        self.mu = (self.a + pro_sum - 1.0) / (self.a + self.b + float(self.model.user_num) - 2)

    def train_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor
    ) -> dict:

        probability_np: np.array \
            = self.exposure_probability[
                  batch_users_tensor.cpu().detach(), batch_items_tensor.cpu().detach()
              ] ** self.expo_weight_exp

        probability: torch.Tensor = torch.Tensor(probability_np).to(self.device)

        # print(self.exposure_probability[batch_users_tensor, batch_items_tensor])
        # print(probability)
        score_loss_mid: torch.Tensor = self.model(batch_users_tensor, batch_items_tensor, batch_scores_tensor)
        # print(score_loss_mid.shape)
        score_loss: torch.Tensor = torch.mean(score_loss_mid * probability)

        L2_reg: torch.Tensor = self.model.get_L2_reg(batch_users_tensor, batch_items_tensor)
        L1_reg: torch.Tensor = self.model.get_L1_reg(batch_users_tensor, batch_items_tensor)

        loss: torch.Tensor = score_loss + L2_reg * self.L2_coe + L1_reg * self.L1_coe

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict: dict = {
            'score_loss': float(score_loss),
            'L2_reg': float(L2_reg),
            'L1_reg': float(L1_reg),
            'loss': float(loss),
        }
        return loss_dict

    def train(self, silent: bool = False, auto: bool = False):
        test_result_list: list = []
        test_epoch_list: list = []

        loss_result_list: list = []
        train_epoch_index_list: list = []

        temp_eval_result: dict = self.evaluator.evaluate()
        test_result_list.append(temp_eval_result)
        test_epoch_list.append(self.epoch_cnt)

        if not silent and not auto:
            print('test at epoch:', self.epoch_cnt)
            print(transfer_loss_dict_to_line_str(temp_eval_result))

        while self.epoch_cnt < self.epochs:
            if (self.epoch_cnt % self.upd_expo_interval) == 0:
                self.calculate_exposure_probability()
            temp_loss_dict = self.train_a_epoch()
            self.upd_mu()
            train_epoch_index_list.append(self.epoch_cnt)
            loss_result_list.append(temp_loss_dict)
            if not silent and not auto:
                print('train epoch:', self.epoch_cnt)
                print(transfer_loss_dict_to_line_str(temp_loss_dict))

            if (self.epoch_cnt % self.evaluate_interval) == 0 and self.epoch_cnt >= self.test_begin_epoch:
                temp_eval_result: dict = self.evaluator.evaluate()
                test_result_list.append(temp_eval_result)
                test_epoch_list.append(self.epoch_cnt)

                if not silent and not auto:
                    print('test at epoch:', self.epoch_cnt)
                    print(transfer_loss_dict_to_line_str(temp_eval_result))

        return (loss_result_list, train_epoch_index_list), \
               (test_result_list, test_epoch_list)


class WMFTrainManager(BasicImplicitTrainManager):
    def __init__(
            self, model: BasicRecommender, evaluator: ImplicitTestManager,
            device: torch.device,
            training_data: torch.Tensor, batch_size: int,
            epochs: int, evaluate_interval: int, lr: float,
            L2_coe: float, L1_coe: float, test_begin_epoch: int = 0,
            imputation_coe: float = 1.0, user_batch_size: int = 1000, item_batch_size: int = 1000
    ):
        super(WMFTrainManager, self).__init__(
            model=model, evaluator=evaluator, device=device, training_data=training_data,
            batch_size=batch_size, epochs=epochs, evaluate_interval=evaluate_interval, lr=lr,
            L2_coe=L2_coe, L1_coe=L1_coe, test_begin_epoch=test_begin_epoch
        )
        self.imputation_coe: float = imputation_coe
        self.zero_tensor: torch.Tensor = torch.Tensor(
            torch.unique(self.users_tensor).shape[0] * torch.unique(self.items_tensor).shape[0]
        ).to(self.device)

        self.user_batch_size: int = user_batch_size
        self.item_batch_size: int = item_batch_size

    def train_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
            *args
    ) -> dict:
        # print()

        score_loss: torch.Tensor = self.model(batch_users_tensor, batch_items_tensor, batch_scores_tensor)
        L2_reg: torch.Tensor = self.model.get_L2_reg(batch_users_tensor, batch_items_tensor)
        L1_reg: torch.Tensor = self.model.get_L1_reg(batch_users_tensor, batch_items_tensor)

        unique_users: torch.Tensor = torch.unique(batch_users_tensor).long()
        unique_items: torch.Tensor = torch.unique(batch_items_tensor).long()

        rand_user_index = np.array(list(range(unique_users.shape[0])))
        np.random.shuffle(rand_user_index)

        rand_item_index = np.array(list(range(unique_items.shape[0])))
        np.random.shuffle(rand_item_index)

        unique_users: torch.Tensor = torch.unique(batch_users_tensor).long()[rand_user_index[0:self.user_batch_size]]
        unique_items: torch.Tensor = torch.unique(batch_items_tensor).long()[rand_item_index[0:self.item_batch_size]]

        imputation_pairs: torch.Tensor = torch.cartesian_prod(unique_users, unique_items)
        imputation_user: torch.Tensor = imputation_pairs[:, 0].reshape(-1)
        imputation_item: torch.Tensor = imputation_pairs[:, 1].reshape(-1)

        # print(imputation_pairs.shape)
        # print(unique_users.shape)
        # print(unique_items.shape)

        imputation_score: torch.Tensor = self.zero_tensor[0:imputation_pairs.shape[0]]
        imputation_loss: torch.Tensor = self.model(imputation_user, imputation_item, imputation_score)

        loss: torch.Tensor \
            = score_loss + L2_reg * self.L2_coe + imputation_loss * self.imputation_coe + L1_reg * self.L1_coe

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict: dict = {
            'score_loss': float(score_loss),
            'L2_reg': float(L2_reg),
            'L1_reg': float(L1_reg),
            'loss': float(loss),
        }
        return loss_dict


class FairnessMFTrainManager(BasicImplicitTrainManager):
    def __init__(
            self, model: BasicRecommender, evaluator: ImplicitTestManager,
            device: torch.device,
            training_data: torch.Tensor, batch_size: int,
            epochs: int, evaluate_interval: int, lr: float,
            L2_coe: float, L1_coe: float, test_begin_epoch: int = 0,
            fairness_coe: float = 1.0, weight_smooth_coe: float = 1.0,
            item_batch_size: int = 1000
    ):
        super(FairnessMFTrainManager, self).__init__(
            model=model, evaluator=evaluator, device=device, training_data=training_data,
            batch_size=batch_size, epochs=epochs, evaluate_interval=evaluate_interval, lr=lr,
            L2_coe=L2_coe, L1_coe=L1_coe, test_begin_epoch=test_begin_epoch
        )
        self.fairness_coe: float = fairness_coe
        self.weight_smooth_coe = weight_smooth_coe
        self.item_distance_tensor: torch.Tensor = self.init_item_distance()
        self.item_batch_size: int = item_batch_size

    def init_item_distance(self):
        items_np: np.array = self.items_tensor.detach().cpu().numpy()
        items_cnt_dict: dict = dict(Counter(items_np))

        max_item_id_size: int = max(items_np) + 1
        items_cnt_array: np.array = np.zeros(max_item_id_size)

        for iid in items_cnt_dict.keys():
            items_cnt_array[iid] = items_cnt_dict[iid]

        item_dis_np_matrix: np.array = np.zeros([max_item_id_size, max_item_id_size])

        max_cnt = max(items_cnt_array)
        min_cnt = min(items_cnt_array)

        max_abs_dis = max_cnt - min_cnt

        for iid_x in range(max_item_id_size):
            for iid_y in range(max_item_id_size):
                pair_dis = np.abs(items_cnt_array[iid_x] - items_cnt_array[iid_y]) / max_abs_dis
                item_dis_np_matrix[iid_x][iid_y] = pair_dis

        item_dis_np_matrix = item_dis_np_matrix / np.max(item_dis_np_matrix)

        item_dis_np_matrix = item_dis_np_matrix ** self.weight_smooth_coe

        return torch.Tensor(item_dis_np_matrix).to(self.device)

    def train_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor
    ) -> dict:
        # print()

        score_loss: torch.Tensor = self.model(batch_users_tensor, batch_items_tensor, batch_scores_tensor)
        L2_reg: torch.Tensor = self.model.get_L2_reg(batch_users_tensor, batch_items_tensor)
        L1_reg: torch.Tensor = self.model.get_L1_reg(batch_users_tensor, batch_items_tensor)

        random_item_index: np.array = np.random.randint(0, self.model.item_num, size=self.item_batch_size)
        user_rating_tensor: torch.Tensor = self.model.predict(batch_users_tensor)[:, random_item_index]

        temp: torch.Tensor = self.item_distance_tensor.detach()[:, random_item_index]
        dis_matrix: torch.Tensor = temp[random_item_index, :]

        temp: torch.Tensor = torch.matmul(torch.matmul(user_rating_tensor, dis_matrix), user_rating_tensor.t())
        fairness_reg: torch.Tensor = torch.trace(temp) / temp.shape[0]

        loss: torch.Tensor \
            = score_loss + L2_reg * self.L2_coe + L1_reg * self.L1_coe + fairness_reg * self.fairness_coe

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict: dict = {
            'score_loss': float(score_loss),
            'L2_reg': float(L2_reg),
            'L1_reg': float(L1_reg),
            'loss': float(loss),
        }
        return loss_dict



class IPSBasicTrainManager(BasicImplicitTrainManager):
    def __init__(
            self, model: BasicRecommender, propensity_func,
            evaluator: ImplicitTestManager,
            device: torch.device,
            training_data: torch.Tensor, batch_size: int,
            epochs: int, evaluate_interval: int, lr: float,
            L2_coe: float, L1_coe: float, test_begin_epoch: int = 0,
            smooth_weight_coe: float = 1.0, uniform_data: torch.Tensor = None
    ):
        super(IPSBasicTrainManager, self).__init__(
            model=model, evaluator=evaluator, device=device, training_data=training_data,
            batch_size=batch_size, epochs=epochs, evaluate_interval=evaluate_interval, lr=lr,
            L2_coe=L2_coe, L1_coe=L1_coe, test_begin_epoch=test_begin_epoch
        )

        self.loss_func: nn.BCELoss = nn.BCELoss(reduction='none')
        self.smooth_weight_coe: float = smooth_weight_coe

        user_inter_cnt_dict: dict = dict(Counter(self.users_tensor.detach().cpu().numpy()))
        item_inter_cnt_dict: dict = dict(Counter(self.items_tensor.detach().cpu().numpy()))

        self.user_inter_cnt_np: np.array = np.zeros(self.model.user_num)
        self.item_inter_cnt_np: np.array = np.zeros(self.model.item_num)

        for uid in user_inter_cnt_dict.keys():
            self.user_inter_cnt_np[uid] = user_inter_cnt_dict[uid]

        for iid in item_inter_cnt_dict.keys():
            self.item_inter_cnt_np[iid] = item_inter_cnt_dict[iid]

        self.user_inter_cnt_np = np.clip(self.user_inter_cnt_np, 1, max(self.user_inter_cnt_np))
        self.item_inter_cnt_np = np.clip(self.item_inter_cnt_np, 1, max(self.item_inter_cnt_np))

        self.propensity_func = propensity_func

        user_np: np.array = self.users_tensor.detach().cpu().numpy().reshape(-1, 1)
        item_np: np.array = self.items_tensor.detach().cpu().numpy().reshape(-1, 1)
        interaction: np.array = np.concatenate([user_np, item_np], axis=1)

        if uniform_data is not None:
            self.uniform_user: torch.Tensor = uniform_data[:, 0].to(self.device).long()
            self.uniform_item: torch.Tensor = uniform_data[:, 1].to(self.device).long()
            self.uniform_score: torch.Tensor = uniform_data[:, 2].to(self.device).float()
            inverse_propensity_np: np.array = self.propensity_func(
                training_data.cpu().detach().numpy(),
                uniform_data.cpu().detach().numpy(),
                self.model.user_num,
                self.model.item_num,
                smooth_weight_coe
            )
        else:
            inverse_propensity_np: np.array = self.propensity_func(
                self.user_inter_cnt_np,
                self.item_inter_cnt_np,
                interaction,
                smooth_weight_coe
            )
        # print(inverse_propensity_np)
        # print(inverse_propensity_np.shape)
        self.inverse_propensity_tensor: torch.Tensor = torch.Tensor(inverse_propensity_np).to(self.device)

    def train_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
            weight_tensor: torch.Tensor = None
    ) -> dict:

        model_pred: torch.Tensor = self.model(batch_users_tensor, batch_items_tensor)
        L2_reg: torch.Tensor = self.model.get_L2_reg(batch_users_tensor, batch_items_tensor)
        L1_reg: torch.Tensor = self.model.get_L1_reg(batch_users_tensor, batch_items_tensor)

        if weight_tensor is None:
            score_loss: torch.Tensor = torch.mean(self.loss_func(model_pred, batch_scores_tensor))
        else:
            score_loss: torch.Tensor = torch.mean(
                self.loss_func(model_pred, batch_scores_tensor) * weight_tensor
            )

        loss: torch.Tensor = score_loss + L2_reg * self.L2_coe + L1_reg * self.L1_coe

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict: dict = {
            'score_loss': float(score_loss),
            'L2_reg': float(L2_reg),
            'L1_reg': float(L1_reg),
            'loss': float(loss),
        }
        return loss_dict

    def train_a_epoch(self) -> dict:
        self.model.train()
        loss_dicts_list: list = []

        for (batch_index, (
                batch_users_tensor, batch_items_tensor, batch_scores_tensor, batch_weight_tensor
        )) in enumerate(mini_batch(
            self.batch_size, self.users_tensor,
            self.items_tensor, self.scores_tensor,
            self.inverse_propensity_tensor
        )):

            loss_dict: dict = self.train_a_batch(
                batch_users_tensor=batch_users_tensor,
                batch_items_tensor=batch_items_tensor,
                batch_scores_tensor=batch_scores_tensor,
                weight_tensor=batch_weight_tensor
            )
            loss_dicts_list.append(loss_dict)

        self.epoch_cnt += 1

        mean_loss_dict: dict = merge_dict(loss_dicts_list, _mean_merge_dict_func)

        return mean_loss_dict


class SNIPSMFTrainManager(IPSBasicTrainManager):
    def __init__(
            self, model: BasicRecommender, propensity_func,
            evaluator: ImplicitTestManager,
            device: torch.device,
            training_data: torch.Tensor, batch_size: int,
            epochs: int, evaluate_interval: int, lr: float,
            L2_coe: float, L1_coe: float, test_begin_epoch: int = 0,
            smooth_weight_coe: float = 1.0, uniform_data: torch.Tensor = None
    ):
        super(SNIPSMFTrainManager, self).__init__(
            model=model, propensity_func=propensity_func,
            evaluator=evaluator, device=device, training_data=training_data,
            batch_size=batch_size, epochs=epochs, evaluate_interval=evaluate_interval, lr=lr,
            L2_coe=L2_coe, L1_coe=L1_coe, test_begin_epoch=test_begin_epoch, smooth_weight_coe=smooth_weight_coe,
            uniform_data=uniform_data
        )

    def train_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
            weight_tensor: torch.Tensor = None
    ) -> dict:

        # print()
        model_pred: torch.Tensor = self.model(batch_users_tensor, batch_items_tensor)
        L2_reg: torch.Tensor = self.model.get_L2_reg(batch_users_tensor, batch_items_tensor)
        L1_reg: torch.Tensor = self.model.get_L1_reg(batch_users_tensor, batch_items_tensor)

        if weight_tensor is None:
            weight_tensor: torch.Tensor = torch.Tensor(np.ones(batch_users_tensor.shape)).float().to(self.device)

        raw_loss: torch.Tensor = self.loss_func(model_pred, batch_scores_tensor)
        weight_loss_sum: torch.Tensor = torch.sum(raw_loss * weight_tensor)
        weight_sum: torch.Tensor = torch.sum(weight_tensor)
        score_loss: torch.Tensor = weight_loss_sum / weight_sum

        loss: torch.Tensor = score_loss + L2_reg * self.L2_coe + L1_reg * self.L1_coe

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict: dict = {
            'score_loss': float(score_loss),
            'L2_reg': float(L2_reg),
            'L1_reg': float(L1_reg),
            'loss': float(loss),
        }
        return loss_dict


def basic_item_propensity_func(
        user_inter_cnt_np: np.array,
        item_inter_cnt_np: np.array,
        interactions: np.array,
        smooth_weight_coe: float
) -> np.array:
    max_item_inter_cnt = max(item_inter_cnt_np)
    item_propensity_np: np.array = (item_inter_cnt_np / max_item_inter_cnt)
    inverse_item_propensity_np: np.array = 1 / item_propensity_np
    item_id_array: np.array = interactions[:, 1].reshape(-1)
    inverse_inter_propensity: np.array = inverse_item_propensity_np[item_id_array] ** smooth_weight_coe

    return inverse_inter_propensity


def basic_user_propensity_func(
        user_inter_cnt_np: np.array,
        item_inter_cnt_np: np.array,
        interactions: np.array,
        smooth_weight_coe: float
) -> np.array:
    max_user_inter_cnt = max(user_inter_cnt_np)
    user_propensity_np: np.array = (user_inter_cnt_np / max_user_inter_cnt)
    inverse_user_propensity_np: np.array = 1 / user_propensity_np
    user_id_array: np.array = interactions[:, 0].reshape(-1)
    inverse_inter_propensity: np.array = inverse_user_propensity_np[user_id_array] ** smooth_weight_coe

    return inverse_inter_propensity


def basic_pair_propensity_func(
        user_inter_cnt_np: np.array,
        item_inter_cnt_np: np.array,
        interactions: np.array,
        smooth_weight_coe: float
) -> np.array:
    max_user_inter_cnt = max(user_inter_cnt_np)
    user_propensity_np: np.array = (user_inter_cnt_np / max_user_inter_cnt)
    inverse_user_propensity_np: np.array = 1 / user_propensity_np

    max_item_inter_cnt = max(item_inter_cnt_np)
    item_propensity_np: np.array = (item_inter_cnt_np / max_item_inter_cnt)
    inverse_item_propensity_np: np.array = 1 / item_propensity_np

    user_id_array: np.array = interactions[:, 0].reshape(-1)
    inverse_inter_propensity_user: np.array = inverse_user_propensity_np[user_id_array]

    item_id_array: np.array = interactions[:, 1].reshape(-1)
    inverse_inter_propensity_item: np.array = inverse_item_propensity_np[item_id_array]

    inverse_inter_propensity \
        = ((inverse_inter_propensity_user + inverse_inter_propensity_item) / 2) ** smooth_weight_coe

    return inverse_inter_propensity


def naive_bayes_propensity(
        train_data: np.array,
        uniform_data: np.array,
        user_num: int,
        item_num: int,
        smooth_weight_coe: float
):
    train_score_np: np.array = train_data[:, 2].reshape(-1)
    uniform_score_np: np.array = uniform_data[:, 2].reshape(-1)
    train_density: float = train_data.shape[0] / (user_num * item_num)
    y_unique: np.array = np.unique(train_score_np)

    p_y_given_0: np.array = np.zeros(y_unique.shape)
    p_y: np.array = np.zeros(y_unique.shape)

    for i in range(len(y_unique)):
        p_y_given_0[i] = np.sum(train_score_np == y_unique[i]) / train_data.shape[0]
        p_y[i] = np.sum(uniform_score_np == y_unique[i]) / uniform_data.shape[0]

    propensity: np.array = p_y_given_0 * train_density / p_y
    propensity = (1 / propensity) ** smooth_weight_coe

    # print(propensity)
    # print(6666)

    weight: np.array = np.zeros(train_score_np.shape[0])

    for i in range(len(y_unique)):
        # print(train_score_np == y_unique[i])
        weight[train_score_np == y_unique[i]] = propensity[i]

    # print(propensity.shape, p_y_given_0.shape, p_y.shape)
    return weight


class CVIBTrainManager(BasicImplicitTrainManager):
    def __init__(
            self, model: BasicRecommender,
            evaluator: ImplicitTestManager,
            device: torch.device,
            training_data: torch.Tensor, batch_size: int,
            epochs: int, evaluate_interval: int, lr: float,
            L2_coe: float, L1_coe: float, test_begin_epoch: int = 0,
            alpha: float = 0.1, gamma: float = 0.01, info_coe: float = 1.0
    ):
        super(CVIBTrainManager, self).__init__(
            model=model, evaluator=evaluator, device=device, training_data=training_data,
            batch_size=batch_size, epochs=epochs, evaluate_interval=evaluate_interval, lr=lr,
            L2_coe=L2_coe, L1_coe=L1_coe, test_begin_epoch=test_begin_epoch
        )
        self.gamma: float = gamma
        self.alpha: float = alpha

        self.loss_func: nn.BCELoss = nn.BCELoss()

        self.info_coe: float = info_coe

    def train_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
            *args
    ) -> dict:

        pred: torch.Tensor = self.model(batch_users_tensor, batch_items_tensor)
        score_loss: torch.Tensor = self.loss_func(pred, batch_scores_tensor)

        rand_user: torch.Tensor \
            = torch.LongTensor(np.random.randint(0, self.model.user_num, batch_scores_tensor.shape[0])).to(self.device)
        rand_item: torch.Tensor \
            = torch.LongTensor(np.random.randint(0, self.model.item_num, batch_scores_tensor.shape[0])).to(self.device)
        # print(rand_user)
        # print(rand_item)
        rand_pred: torch.Tensor = self.model(rand_user, rand_item)

        logp_hat: torch.Tensor = pred.log()
        pred_avg = pred.mean()
        pred_ul_avg = rand_pred.mean()

        info_loss = self.alpha * (- pred_avg * pred_ul_avg.log() - (1 - pred_avg) * (
                    1 - pred_ul_avg).log()) + self.gamma * torch.mean(pred * logp_hat)

        L2_reg: torch.Tensor = self.model.get_L2_reg(batch_users_tensor, batch_items_tensor)
        L1_reg: torch.Tensor = self.model.get_L1_reg(batch_users_tensor, batch_items_tensor)

        loss: torch.Tensor = score_loss + info_loss * self.info_coe + L2_reg * self.L2_coe + L1_reg * self.L1_coe

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict: dict = {
            'score_loss': float(score_loss),
            'L2_reg': float(L2_reg),
            'L1_reg': float(L1_reg),
            'loss': float(loss),
        }
        return loss_dict


class CausETrainManager(BasicUniformImplicitTrainManager):
    def __init__(
            self, model: CausEMatrixFactorization,
            evaluator: ImplicitTestManager,
            device: torch.device,
            training_data: torch.Tensor, uniform_data: torch.Tensor, batch_size: int,
            epochs: int, evaluate_interval: int, lr: float,
            L2_coe: float, L1_coe: float, test_begin_epoch: int = 0,
            uniform_loss_coe: float = 1.0,
            teacher_reg_coe: float = 1.0, teacher_reg_mode: str = 'i',
            teacher_L2_coe: float = 5.
    ):
        super(CausETrainManager, self).__init__(
            model=model, evaluator=evaluator, device=device, training_data=training_data, uniform_data=uniform_data,
            batch_size=batch_size, epochs=epochs, evaluate_interval=evaluate_interval, lr=lr,
            L2_coe=L2_coe, L1_coe=L1_coe, test_begin_epoch=test_begin_epoch
        )
        assert teacher_reg_mode in ['i', 'u', 'ui']
        self.teacher_reg_coe: float = teacher_reg_coe
        self.teacher_reg_mode: str = teacher_reg_mode
        self.model: CausEMatrixFactorization = model
        self.uniform_loss_coe: float = uniform_loss_coe
        self.teacher_L2_coe = teacher_L2_coe

    def train_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
            *args
    ) -> dict:

        # print()

        train_score_loss: torch.Tensor = self.model(
            users_id=batch_users_tensor,
            items_id=batch_items_tensor,
            train_teacher=False,
            ground_truth=batch_scores_tensor
        )

        uniform_score_loss: torch.Tensor = self.model(
            users_id=self.uniform_user,
            items_id=self.uniform_item,
            train_teacher=True,
            ground_truth=self.uniform_score
        )
        L2_reg: torch.Tensor \
            = self.model.get_L2_reg(batch_users_tensor, batch_items_tensor, False) * self.L2_coe + \
              self.model.get_L2_reg(self.uniform_user, self.uniform_item, True) * self.teacher_L2_coe
        teacher_reg: torch.Tensor = torch.Tensor([0.0]).to(self.device)
        if 'i' in self.teacher_reg_mode:
            teacher_reg = teacher_reg + self.model.item_teacher_reg(items_id=batch_items_tensor)

        if 'u' in self.teacher_reg_mode:
            teacher_reg = teacher_reg + self.model.user_teacher_reg(users_id=batch_users_tensor)

        loss: torch.Tensor = train_score_loss \
                             + uniform_score_loss * self.uniform_loss_coe \
                             +  L2_reg + teacher_reg * self.teacher_reg_coe

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict: dict = {
            'train_score_loss': float(train_score_loss),
            'uniform_score_loss': float(uniform_score_loss),
            'teacher_reg': float(teacher_reg),
            'L2_reg': float(L2_reg),
            'loss': float(loss),
        }
        return loss_dict


class CausEExplicitTrainManager(BasicUniformExplicitTrainManager):
    def __init__(
            self, model: CausEExplicitMatrixFactorization,
            evaluator: ExplicitTestManager,
            device: torch.device,
            training_data: torch.Tensor, uniform_data: torch.Tensor, batch_size: int,
            epochs: int, evaluate_interval: int, lr: float,
            L2_coe: float, L1_coe: float, test_begin_epoch: int = 0,
            uniform_loss_coe: float = 1.0,
            teacher_reg_coe: float = 1.0, teacher_reg_mode: str = 'i',
            teacher_L2_coe: float = 5.
    ):
        super(CausEExplicitTrainManager, self).__init__(
            model=model, evaluator=evaluator, device=device, training_data=training_data, uniform_data=uniform_data,
            batch_size=batch_size, epochs=epochs, evaluate_interval=evaluate_interval, lr=lr,
            L2_coe=L2_coe, L1_coe=L1_coe, test_begin_epoch=test_begin_epoch
        )
        assert teacher_reg_mode in ['i', 'u', 'ui']
        self.teacher_reg_coe: float = teacher_reg_coe
        self.teacher_reg_mode: str = teacher_reg_mode
        self.model: CausEExplicitMatrixFactorization = model
        self.uniform_loss_coe: float = uniform_loss_coe
        self.teacher_L2_coe = teacher_L2_coe

    def train_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
            *args
    ) -> dict:

        # print()

        train_score_loss: torch.Tensor = self.model(
            users_id=batch_users_tensor,
            items_id=batch_items_tensor,
            train_teacher=False,
            ground_truth=batch_scores_tensor
        )

        uniform_score_loss: torch.Tensor = self.model(
            users_id=self.uniform_user,
            items_id=self.uniform_item,
            train_teacher=True,
            ground_truth=self.uniform_score
        )
        L2_reg: torch.Tensor \
            = self.model.get_L2_reg(batch_users_tensor, batch_items_tensor, False) * self.L2_coe + \
              self.model.get_L2_reg(self.uniform_user, self.uniform_item, True) * self.teacher_L2_coe
        teacher_reg: torch.Tensor = torch.Tensor([0.0]).to(self.device)
        if 'i' in self.teacher_reg_mode:
            teacher_reg = teacher_reg + self.model.item_teacher_reg(items_id=batch_items_tensor)

        if 'u' in self.teacher_reg_mode:
            teacher_reg = teacher_reg + self.model.user_teacher_reg(users_id=batch_users_tensor)

        loss: torch.Tensor = train_score_loss \
                             + uniform_score_loss * self.uniform_loss_coe \
                             + L2_reg + teacher_reg * self.teacher_reg_coe

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict: dict = {
            'train_score_loss': float(train_score_loss),
            'uniform_score_loss': float(uniform_score_loss),
            'teacher_reg': float(teacher_reg),
            'L2_reg': float(L2_reg),
            'loss': float(loss),
        }
        return loss_dict


class IPSBasicExplicitTrainManager(BasicExplicitTrainManager):
    def __init__(
            self, model: BasicExplicitRecommender, propensity_func,
            evaluator: ExplicitTestManager,
            device: torch.device,
            training_data: torch.Tensor, batch_size: int,
            epochs: int, evaluate_interval: int, lr: float,
            L2_coe: float, L1_coe: float, test_begin_epoch: int = 0,
            smooth_weight_coe: float = 1.0, uniform_data: torch.Tensor = None
    ):
        super(IPSBasicExplicitTrainManager, self).__init__(
            model=model, evaluator=evaluator, device=device, training_data=training_data,
            batch_size=batch_size, epochs=epochs, evaluate_interval=evaluate_interval, lr=lr,
            L2_coe=L2_coe, L1_coe=L1_coe, test_begin_epoch=test_begin_epoch
        )

        self.loss_func: nn.MSELoss = nn.MSELoss(reduction='none')
        self.smooth_weight_coe: float = smooth_weight_coe

        user_inter_cnt_dict: dict = dict(Counter(self.users_tensor.detach().cpu().numpy()))
        item_inter_cnt_dict: dict = dict(Counter(self.items_tensor.detach().cpu().numpy()))

        self.user_inter_cnt_np: np.array = np.zeros(self.model.user_num)
        self.item_inter_cnt_np: np.array = np.zeros(self.model.item_num)

        for uid in user_inter_cnt_dict.keys():
            self.user_inter_cnt_np[uid] = user_inter_cnt_dict[uid]

        for iid in item_inter_cnt_dict.keys():
            self.item_inter_cnt_np[iid] = item_inter_cnt_dict[iid]

        self.user_inter_cnt_np = np.clip(self.user_inter_cnt_np, 1, max(self.user_inter_cnt_np))
        self.item_inter_cnt_np = np.clip(self.item_inter_cnt_np, 1, max(self.item_inter_cnt_np))

        self.propensity_func = propensity_func

        user_np: np.array = self.users_tensor.detach().cpu().numpy().reshape(-1, 1)
        item_np: np.array = self.items_tensor.detach().cpu().numpy().reshape(-1, 1)
        interaction: np.array = np.concatenate([user_np, item_np], axis=1)

        if uniform_data is not None:
            self.uniform_user: torch.Tensor = uniform_data[:, 0].to(self.device).long()
            self.uniform_item: torch.Tensor = uniform_data[:, 1].to(self.device).long()
            self.uniform_score: torch.Tensor = uniform_data[:, 2].to(self.device).float()
            inverse_propensity_np: np.array = self.propensity_func(
                training_data.cpu().detach().numpy(),
                uniform_data.cpu().detach().numpy(),
                self.model.user_num,
                self.model.item_num,
                smooth_weight_coe
            )
        else:
            inverse_propensity_np: np.array = self.propensity_func(
                self.user_inter_cnt_np,
                self.item_inter_cnt_np,
                interaction,
                smooth_weight_coe
            )
        # print(inverse_propensity_np)
        # print(inverse_propensity_np.shape)
        self.inverse_propensity_tensor: torch.Tensor = torch.Tensor(inverse_propensity_np).to(self.device)
        print(self.inverse_propensity_tensor)
        print(2333)

    def train_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
            weight_tensor: torch.Tensor = None
    ) -> dict:

        model_pred: torch.Tensor = self.model(batch_users_tensor, batch_items_tensor)
        L2_reg: torch.Tensor = self.model.get_L2_reg(batch_users_tensor, batch_items_tensor)
        L1_reg: torch.Tensor = self.model.get_L1_reg(batch_users_tensor, batch_items_tensor)

        if weight_tensor is None:
            score_loss: torch.Tensor = torch.mean(self.loss_func(model_pred, batch_scores_tensor))
        else:
            score_loss: torch.Tensor = torch.mean(
                self.loss_func(model_pred, batch_scores_tensor) * weight_tensor
            )

        loss: torch.Tensor = score_loss + L2_reg * self.L2_coe + L1_reg * self.L1_coe

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict: dict = {
            'score_loss': float(score_loss),
            'L2_reg': float(L2_reg),
            'L1_reg': float(L1_reg),
            'loss': float(loss),
        }
        return loss_dict

    def train_a_epoch(self) -> dict:
        self.model.train()
        loss_dicts_list: list = []

        for (batch_index, (
                batch_users_tensor, batch_items_tensor, batch_scores_tensor, batch_weight_tensor
        )) in enumerate(mini_batch(
            self.batch_size, self.users_tensor,
            self.items_tensor, self.scores_tensor,
            self.inverse_propensity_tensor
        )):

            loss_dict: dict = self.train_a_batch(
                batch_users_tensor=batch_users_tensor,
                batch_items_tensor=batch_items_tensor,
                batch_scores_tensor=batch_scores_tensor,
                weight_tensor=batch_weight_tensor
            )
            loss_dicts_list.append(loss_dict)

        self.epoch_cnt += 1

        mean_loss_dict: dict = merge_dict(loss_dicts_list, _mean_merge_dict_func)

        return mean_loss_dict


class SNIPSExplicitMFTrainManager(IPSBasicExplicitTrainManager):
    def __init__(
            self, model: BasicExplicitRecommender, propensity_func,
            evaluator: ExplicitTestManager,
            device: torch.device,
            training_data: torch.Tensor, batch_size: int,
            epochs: int, evaluate_interval: int, lr: float,
            L2_coe: float, L1_coe: float, test_begin_epoch: int = 0,
            smooth_weight_coe: float = 1.0, uniform_data: torch.Tensor = None
    ):
        super(SNIPSExplicitMFTrainManager, self).__init__(
            model=model, propensity_func=propensity_func,
            evaluator=evaluator, device=device, training_data=training_data,
            batch_size=batch_size, epochs=epochs, evaluate_interval=evaluate_interval, lr=lr,
            L2_coe=L2_coe, L1_coe=L1_coe, test_begin_epoch=test_begin_epoch, smooth_weight_coe=smooth_weight_coe,
            uniform_data=uniform_data
        )

    def train_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
            weight_tensor: torch.Tensor = None
    ) -> dict:

        # print()
        model_pred: torch.Tensor = self.model(batch_users_tensor, batch_items_tensor)
        L2_reg: torch.Tensor = self.model.get_L2_reg(batch_users_tensor, batch_items_tensor)
        L1_reg: torch.Tensor = self.model.get_L1_reg(batch_users_tensor, batch_items_tensor)

        if weight_tensor is None:
            weight_tensor: torch.Tensor = torch.Tensor(np.ones(batch_users_tensor.shape)).float().to(self.device)

        raw_loss: torch.Tensor = self.loss_func(model_pred, batch_scores_tensor)
        weight_loss_sum: torch.Tensor = torch.sum(raw_loss * weight_tensor)
        weight_sum: torch.Tensor = torch.sum(weight_tensor)
        score_loss: torch.Tensor = weight_loss_sum / weight_sum

        loss: torch.Tensor = score_loss + L2_reg * self.L2_coe + L1_reg * self.L1_coe

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict: dict = {
            'score_loss': float(score_loss),
            'L2_reg': float(L2_reg),
            'L1_reg': float(L1_reg),
            'loss': float(loss),
        }
        return loss_dict


class CVIBExplicitTrainManager(BasicExplicitTrainManager):
    def __init__(
            self, model: BasicExplicitRecommender,
            evaluator: ExplicitTestManager,
            device: torch.device,
            training_data: torch.Tensor, batch_size: int,
            epochs: int, evaluate_interval: int, lr: float,
            L2_coe: float, L1_coe: float, test_begin_epoch: int = 0,
            alpha: float = 0.1, gamma: float = 0.01, info_coe: float = 1.0,
            eps: float = 1e-1
    ):
        super(CVIBExplicitTrainManager, self).__init__(
            model=model, evaluator=evaluator, device=device, training_data=training_data,
            batch_size=batch_size, epochs=epochs, evaluate_interval=evaluate_interval, lr=lr,
            L2_coe=L2_coe, L1_coe=L1_coe, test_begin_epoch=test_begin_epoch
        )
        self.gamma: float = gamma
        self.alpha: float = alpha

        self.loss_func: nn.MSELoss = nn.MSELoss()

        self.info_coe: float = info_coe
        self.eps: float = eps

    def train_a_batch(
            self,
            batch_users_tensor: torch.Tensor,
            batch_items_tensor: torch.Tensor,
            batch_scores_tensor: torch.Tensor,
            *args
    ) -> dict:

        pred: torch.Tensor = self.model(batch_users_tensor, batch_items_tensor)
        score_loss: torch.Tensor = self.loss_func(pred, batch_scores_tensor)

        rand_user: torch.Tensor \
            = torch.LongTensor(np.random.randint(0, self.model.user_num, batch_scores_tensor.shape[0])).to(self.device)
        rand_item: torch.Tensor \
            = torch.LongTensor(np.random.randint(0, self.model.item_num, batch_scores_tensor.shape[0])).to(self.device)
        # print(rand_user)
        # print(rand_item)
        rand_pred: torch.Tensor = self.model(rand_user, rand_item)

        logp_hat: torch.Tensor = torch.clip(pred, min=self.eps).log()
        pred_avg = pred.mean()
        pred_ul_avg = rand_pred.mean()

        info_loss = self.alpha * (- pred_avg * torch.clip(pred_ul_avg, min=self.eps).log() - (1 - pred_avg) *
                                  torch.clip(1 - pred_ul_avg, min=self.eps).log()) \
                    + self.gamma * torch.mean(pred * logp_hat)

        L2_reg: torch.Tensor = self.model.get_L2_reg(batch_users_tensor, batch_items_tensor)
        L1_reg: torch.Tensor = self.model.get_L1_reg(batch_users_tensor, batch_items_tensor)

        loss: torch.Tensor = score_loss + info_loss * self.info_coe + L2_reg * self.L2_coe + L1_reg * self.L1_coe

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict: dict = {
            'score_loss': float(score_loss),
            'L2_reg': float(L2_reg),
            'L1_reg': float(L1_reg),
            'loss': float(loss),
        }
        return loss_dict
