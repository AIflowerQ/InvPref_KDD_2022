import pandas as pd
import numpy as np
from datetime import datetime
import torch
from tqdm import tqdm

from utils import analyse_interaction_from_text, analyse_user_interacted_set


class BaseImplicitDataLoader:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def user_mask_items(self, user_id: int) -> set:
        raise NotImplementedError

    def user_highlight_items(self, user_id: int) -> set:
        raise NotImplementedError

    @property
    def all_test_users_by_sorted_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def all_test_users_by_sorted_list(self) -> list:
        raise NotImplementedError

    def get_user_ground_truth(self, user_id: int) -> set:
        raise NotImplementedError

    @property
    def get_sorted_all_test_users_ground_truth(self) -> list:
        raise NotImplementedError

    @property
    def train_data_len(self) -> int:
        raise NotImplementedError

    @property
    def test_data_len(self) -> int:
        raise NotImplementedError

    @property
    def user_num(self) -> int:
        raise NotImplementedError

    @property
    def item_num(self) -> int:
        raise NotImplementedError

    @property
    def test_data_df(self) -> pd.DataFrame:
        raise NotImplementedError

    @property
    def test_data_np(self) -> np.array:
        raise NotImplementedError


class BaseImplicitBCELossDataLoader(BaseImplicitDataLoader):
    def __init__(self, dataset_path: str):
        super(BaseImplicitBCELossDataLoader, self).__init__(dataset_path)

    def user_mask_items(self, user_id: int) -> set:
        raise NotImplementedError

    def user_highlight_items(self, user_id: int) -> set:
        raise NotImplementedError

    @property
    def all_test_users_by_sorted_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def all_test_users_by_sorted_list(self) -> list:
        raise NotImplementedError

    def get_user_ground_truth(self, user_id: int) -> set:
        raise NotImplementedError

    @property
    def get_sorted_all_test_users_ground_truth(self) -> list:
        raise NotImplementedError

    @property
    def train_data_len(self) -> int:
        raise NotImplementedError

    @property
    def test_data_len(self) -> int:
        raise NotImplementedError

    @property
    def user_num(self) -> int:
        raise NotImplementedError

    @property
    def item_num(self) -> int:
        raise NotImplementedError

    @property
    def test_data_df(self) -> pd.DataFrame:
        raise NotImplementedError

    @property
    def train_data_df(self) -> pd.DataFrame:
        raise NotImplementedError

    @property
    def test_data_np(self) -> np.array:
        raise NotImplementedError

    @property
    def train_data_np(self) -> np.array:
        raise NotImplementedError


class YahooImplicitBCELossDataLoader(BaseImplicitBCELossDataLoader):
    def __init__(self, dataset_path: str, device: torch.device, has_item_pool_file: bool = False):
        super(YahooImplicitBCELossDataLoader, self).__init__(dataset_path)
        self.train_data_path: str = self.dataset_path + '/train.csv'
        self.test_data_path: str = self.dataset_path + '/test.csv'

        self.train_df: pd.DataFrame = pd.read_csv(self.train_data_path)  # [0: 100000]
        self.test_df: pd.DataFrame = pd.read_csv(self.test_data_path)

        self._train_data: np.array = self.train_df.values.astype(np.int64)
        self._test_data: np.array = self.test_df.values.astype(np.int64)

        self.user_positive_interaction = []
        self.user_list: list = []
        self.item_list: list = []

        self._user_num = 0
        self._item_num = 0

        self.test_user_list: list = []
        self.test_item_list: list = []
        self.ground_truth: list = []

        self.has_item_pool: bool = has_item_pool_file

        with open(self.train_data_path, 'r') as inp:
            inp.readline()
            lines: list = inp.readlines()

            print('Begin analyze raw train file')
            pairs, self.user_list, self.item_list = analyse_interaction_from_text(lines, has_value=True)

            positive_pairs: list = list(filter(lambda pair: pair[2] > 0, pairs))

            user_positive_interaction: list = analyse_user_interacted_set(positive_pairs)
            self.user_positive_interaction = user_positive_interaction

            self._train_pairs: list = pairs

            inp.close()

        with open(self.test_data_path, 'r') as inp:
            inp.readline()
            lines: list = inp.readlines()
            print('Begin analyze raw test file')
            pairs, self.test_user_list, self.test_item_list = analyse_interaction_from_text(lines)
            # print(self.test_user_list)
            self.ground_truth: list = analyse_user_interacted_set(pairs)
            inp.close()

        if self.has_item_pool:
            self.item_pool_path: str = self.dataset_path + '/test_item_pool.csv'
            with open(self.item_pool_path, 'r') as inp:
                inp.readline()
                lines: list = inp.readlines()
                print('Begin analyze item pool file')
                pairs, _, _ = analyse_interaction_from_text(lines)

                self.item_pool: list = analyse_user_interacted_set(pairs)
                inp.close()

        self._user_num = max(self.user_list + self.test_user_list) + 1
        self._item_num = max(self.item_list + self.test_item_list) + 1

        self.test_users_tensor: torch.LongTensor = torch.LongTensor(self.test_user_list)
        self.test_users_tensor = self.test_users_tensor.to(device)
        self.sorted_ground_truth: list = [self.get_user_ground_truth(user_id) for user_id in self.test_user_list]

    def user_mask_items(self, user_id: int) -> set:
        return self.user_positive_interaction[user_id]

    def user_highlight_items(self, user_id: int) -> set:
        if not self.has_item_pool:
            raise NotImplementedError('Not has item pool!')
        return self.item_pool[user_id]

    @property
    def all_test_users_by_sorted_tensor(self) -> torch.Tensor:
        """
        注意，这个tensor没有指明device
        :return:
        """
        return self.test_users_tensor

    @property
    def all_test_users_by_sorted_list(self) -> list:
        return self.test_user_list

    def get_user_ground_truth(self, user_id: int) -> set:
        return self.ground_truth[user_id]

    @property
    def get_sorted_all_test_users_ground_truth(self) -> list:
        return self.sorted_ground_truth

    @property
    def train_data_len(self) -> int:
        return self.train_df.shape[0]

    @property
    def test_data_len(self) -> int:
        return self.test_df.shape[0]

    @property
    def user_num(self) -> int:
        return self._user_num

    @property
    def item_num(self) -> int:
        return self._item_num

    @property
    def test_data_df(self) -> pd.DataFrame:
        return self.test_df

    @property
    def train_data_df(self) -> pd.DataFrame:
        return self.train_df

    @property
    def test_data_np(self) -> np.array:
        return self._test_data

    @property
    def train_data_np(self) -> np.array:
        return self._train_data


class YahooUniformImplicitBCELossDataLoader(YahooImplicitBCELossDataLoader):
    def __init__(self, dataset_path: str, device: torch.device, has_item_pool_file: bool = False):
        super(YahooUniformImplicitBCELossDataLoader, self).__init__(
            dataset_path,
            device,
            has_item_pool_file
        )
        self.uniform_data_path: str = self.dataset_path + '/uniform_train.csv'
        self.uniform_df: pd.DataFrame = pd.read_csv(self.uniform_data_path)
        self._uniform_data: np.array = self.uniform_df.values.astype(np.int64)

    @property
    def uniform_data_np(self) -> np.array:
        return self._uniform_data

    @property
    def uniform_data_len(self) -> int:
        return self.uniform_df.shape[0]


class ImplicitBCELossDataLoaderStaticPopularity(YahooImplicitBCELossDataLoader):
    def __init__(self, dataset_path: str, device: torch.device, has_item_pool_file: bool = False):
        super(ImplicitBCELossDataLoaderStaticPopularity, self).__init__(
            dataset_path,
            device,
            has_item_pool_file
        )

        self.user_inter_cnt_np: np.array = np.zeros(self.user_num).astype(np.int64)
        self.item_inter_cnt_np: np.array = np.zeros(self.item_num).astype(np.int64)

        for pair in self._train_pairs:
            uid, iid = pair[0], pair[1]
            self.user_inter_cnt_np[uid] += 1
            self.item_inter_cnt_np[iid] += 1

        self.max_user_inter_cnt = self.user_inter_cnt_np.max()
        self.min_user_inter_cnt = self.user_inter_cnt_np.min()
        self.max_item_inter_cnt = self.item_inter_cnt_np.max()
        self.min_item_inter_cnt = self.item_inter_cnt_np.min()

        self.user_inter_cnt_normalize_np: np.array \
            = (self.user_inter_cnt_np - self.min_user_inter_cnt) / (self.max_user_inter_cnt - self.min_user_inter_cnt)

        self.item_inter_cnt_normalize_np: np.array \
            = (self.item_inter_cnt_np - self.min_item_inter_cnt) / (self.max_item_inter_cnt - self.min_item_inter_cnt)

    def query_users_inter_cnt(self, users_id):
        return self.user_inter_cnt_np[users_id]

    def query_items_inter_cnt(self, items_id):
        return self.item_inter_cnt_np[items_id]

    def query_users_inter_cnt_normalize(self, users_id):
        return self.user_inter_cnt_normalize_np[users_id]

    def query_items_inter_cnt_normalize(self, items_id):
        return self.item_inter_cnt_normalize_np[items_id]

    def query_pairs_cnt_add(self, users_id, items_id):
        users_cnt: np.array = self.user_inter_cnt_np[users_id]
        items_cnt: np.array = self.item_inter_cnt_np[items_id]

        return users_cnt + items_cnt

    def query_pairs_cnt_normalize_multiply(self, users_id, items_id):
        users_cnt_normalize: np.array = self.user_inter_cnt_normalize_np[users_id]
        items_cnt_normalize: np.array = self.item_inter_cnt_normalize_np[items_id]

        return users_cnt_normalize * items_cnt_normalize


class BaseExplicitDataLoader:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    @property
    def all_test_pairs_np(self) -> np.array:
        raise NotImplementedError

    @property
    def all_test_scores_np(self) -> np.array:
        raise NotImplementedError

    @property
    def test_data_np(self) -> np.array:
        raise NotImplementedError

    @property
    def all_test_pairs_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def all_test_scores_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def test_data_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def all_train_pairs_np(self) -> np.array:
        raise NotImplementedError

    @property
    def all_train_scores_np(self) -> np.array:
        raise NotImplementedError

    @property
    def train_data_np(self) -> np.array:
        raise NotImplementedError

    @property
    def all_train_pairs_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def all_train_scores_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def train_data_tensor(self) -> torch.Tensor:
        raise NotImplementedError


    @property
    def train_data_len(self) -> int:
        raise NotImplementedError

    @property
    def test_data_len(self) -> int:
        raise NotImplementedError

    @property
    def user_num(self) -> int:
        raise NotImplementedError

    @property
    def item_num(self) -> int:
        raise NotImplementedError


class ExplicitDataLoader(BaseExplicitDataLoader):
    def __init__(self, dataset_path: str, device: torch.device):
        super(ExplicitDataLoader, self).__init__(dataset_path)
        self.device: torch.device = device
        self.train_data_path: str = self.dataset_path + '/train.csv'
        self.test_data_path: str = self.dataset_path + '/test.csv'

        self.train_df: pd.DataFrame = pd.read_csv(self.train_data_path)  # [0: 100000]
        self.test_df: pd.DataFrame = pd.read_csv(self.test_data_path)

        self._train_data: np.array = self.train_df.values.astype(np.int64)
        self._test_data: np.array = self.test_df.values.astype(np.int64)

        self._train_data_tensor: torch.Tensor = torch.LongTensor(self._train_data).to(self.device)
        self._test_data_tensor: torch.Tensor = torch.LongTensor(self._test_data).to(self.device)

        self.user_positive_interaction = []

        self._user_num = int(np.max(self._train_data[:, 0].reshape(-1))) + 1
        self._item_num = int(np.max(self._train_data[:, 1].reshape(-1))) + 1

        self._train_pairs: np.array = self._train_data[:, 0:2].astype(np.int64).reshape(-1, 2)
        self._test_pairs: np.array = self._test_data[:, 0:2].astype(np.int64).reshape(-1, 2)

        self._train_pairs_tensor: torch.Tensor = torch.LongTensor(self._train_pairs).to(self.device)
        self._test_pairs_tensor: torch.Tensor = torch.LongTensor(self._test_pairs).to(self.device)

        self._train_scores: np.array = self._train_data[:, 2].astype(np.float64).reshape(-1)
        self._test_scores: np.array = self._test_data[:, 2].astype(np.float64).reshape(-1)

        self._train_scores_tensor: torch.Tensor = torch.Tensor(self._train_scores).to(self.device)
        self._test_scores_tensor: torch.Tensor = torch.Tensor(self._test_scores).to(self.device)

    @property
    def all_test_pairs_np(self) -> np.array:
        return self._test_pairs

    @property
    def all_test_scores_np(self) -> np.array:
        return self._test_scores

    @property
    def test_data_np(self) -> np.array:
        return self._test_data

    @property
    def all_train_pairs_np(self) -> np.array:
        return self._train_pairs

    @property
    def all_train_scores_np(self) -> np.array:
        return self._train_scores

    @property
    def train_data_np(self) -> np.array:
        return self._train_data

    @property
    def train_data_len(self) -> int:
        return self._train_data.shape[0]

    @property
    def test_data_len(self) -> int:
        return self._test_data.shape[0]

    @property
    def user_num(self) -> int:
        return self._user_num

    @property
    def item_num(self) -> int:
        return self._item_num

    @property
    def all_test_pairs_tensor(self) -> torch.Tensor:
        return self._test_pairs_tensor

    @property
    def all_test_scores_tensor(self) -> torch.Tensor:
        return self._test_scores_tensor

    @property
    def test_data_tensor(self) -> torch.Tensor:
        return self._test_data_tensor

    @property
    def all_train_pairs_tensor(self) -> torch.Tensor:
        return self._train_pairs_tensor

    @property
    def all_train_scores_tensor(self) -> torch.Tensor:
        return self._train_scores_tensor

    @property
    def train_data_tensor(self) -> torch.Tensor:
        return self._train_data_tensor


class ExplicitUniformDataLoader(ExplicitDataLoader):
    def __init__(self, dataset_path: str, device: torch.device):
        super(ExplicitUniformDataLoader, self).__init__(dataset_path, device)
        self.uniform_data_path: str = self.dataset_path + '/uniform_train.csv'
        self.uniform_df: pd.DataFrame = pd.read_csv(self.uniform_data_path)
        self._uniform_data: np.array = self.uniform_df.values.astype(np.int64)

    @property
    def uniform_data_np(self) -> np.array:
        return self._uniform_data

    @property
    def uniform_data_len(self) -> int:
        return self.uniform_df.shape[0]

