import json

import global_config
from dataloader import ExplicitDataLoader
import torch
import numpy as np

from evaluate import ExplicitTestManager
from models import InvPrefExplicit
from train import ExplicitTrainManager
from utils import draw_score_pic, merge_dict, _show_me_a_list_func, draw_loss_pic_one_by_one, query_user, query_str, \
    mkdir, query_int, get_class_name_str, show_me_all_the_fucking_explicit_result, _mean_merge_dict_func

torch.cuda.set_device(3)
DEVICE: torch.device = torch.device('cuda')

MODEL_CONFIG: dict = {
    'env_num': 5,
    'factor_num': 40,
    'reg_only_embed': True,
    'reg_env_embed': False
}

TRAIN_CONFIG: dict = {
    "batch_size": 131072,
    "epochs": 1000,
    "cluster_interval": 20,
    "evaluate_interval": 10,
    "lr": 1e-3,
    "invariant_coe": 0.007375309563638757,
    "env_aware_coe": 7.207790368836971,
    "env_coe": 7.30272189219841,
    "L2_coe": 5.105587170019545,
    "L1_coe": 0.004098813161410509,
    "alpha": None,
    "use_class_re_weight": False,
    "use_recommend_re_weight": False,
    'test_begin_epoch': 0,
    'begin_cluster_epoch': None,
    'stop_cluster_epoch': None,
}

EVALUATE_CONFIG: dict = {
    'eval_metric': 'mse'
}

RANDOM_SEED_LIST = [17373331, 17373511, 17373423]

DATASET_PATH = '/Yahoo_explicit_all_data/'
METRIC_LIST = ['mse', 'rmse', 'mae']


def main(
        device: torch.device,
        model_config: dict,
        train_config: dict,
        evaluate_config: dict,
        data_loader: ExplicitDataLoader,
        random_seed: int,
        silent: bool = False,
        auto: bool = False,
        query: bool = True,
):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)

    model: InvPrefExplicit \
        = InvPrefExplicit(
        user_num=data_loader.user_num,
        item_num=data_loader.item_num,
        env_num=model_config['env_num'],
        factor_num=model_config['factor_num'],
        reg_only_embed=model_config['reg_only_embed'],
        reg_env_embed=model_config['reg_env_embed']
    )
    model = model.to(device)

    evaluator: ExplicitTestManager = ExplicitTestManager(
        model=model,
        data_loader=data_loader
    )

    train_tensor: torch.LongTensor = torch.LongTensor(data_loader.train_data_np).to(device)

    print(train_tensor.shape)
    assert train_tensor.shape[1] == 3

    train_manager: ExplicitTrainManager = ExplicitTrainManager(
        model=model, evaluator=evaluator, training_data=train_tensor,
        device=device,
        batch_size=train_config['batch_size'], epochs=train_config['epochs'],
        cluster_interval=train_config['cluster_interval'],
        evaluate_interval=train_config['evaluate_interval'],
        lr=train_config['lr'], invariant_coe=train_config['invariant_coe'],
        env_aware_coe=train_config['env_aware_coe'], env_coe=train_config['env_coe'],
        L2_coe=train_config['L2_coe'], L1_coe=train_config['L1_coe'], alpha=train_config['alpha'],
        use_class_re_weight=train_config['use_class_re_weight'],
        test_begin_epoch=train_config['test_begin_epoch'], begin_cluster_epoch=train_config['begin_cluster_epoch'],
        stop_cluster_epoch=train_config['stop_cluster_epoch'],
        use_recommend_re_weight=train_config['use_recommend_re_weight']
    )

    train_tuple, test_tuple, cluster_tuple = train_manager.train(silent=silent, auto=auto)

    test_result_list = test_tuple[0]
    result_merged_by_metric: dict = merge_dict(test_result_list, _show_me_a_list_func)
    eval_metric: str = evaluate_config['eval_metric']

    stand_result: np.array = np.array(result_merged_by_metric[eval_metric])

    best_performance = np.min(stand_result)
    best_indexes: list = np.where(stand_result == best_performance)[0].tolist()
    fucking_result: dict = show_me_all_the_fucking_explicit_result(
        result_merged_by_metric,
        METRIC_LIST,
        best_indexes[0]
    )
    if not auto:
        print('Best {}:'.format(eval_metric), best_performance, best_indexes)

    if not silent and not auto:
        loss_dict_list: list = train_tuple[0]

        loss_dict: dict = merge_dict(loss_dict_list, _show_me_a_list_func)
        draw_loss_pic_one_by_one(len(train_tuple[1]), **loss_dict)

        # all_test_result: dict = {}

        draw_score_pic(test_tuple[1], **result_merged_by_metric)

        draw_score_pic(cluster_tuple[2], diff_num=cluster_tuple[0])

        dict_to_show: dict = dict()
        temp_dict = merge_dict(cluster_tuple[1], _show_me_a_list_func)
        for key in temp_dict.keys():
            dict_to_show[str(key)] = temp_dict[key]
        draw_score_pic(cluster_tuple[2], **dict_to_show)

    result_json: str = json.dumps(result_merged_by_metric, indent=4)
    evaluate_config_json: str = json.dumps(evaluate_config, indent=4)
    model_config_json: str = json.dumps(model_config, indent=4)
    train_config_json: str = json.dumps(train_config, indent=4)

    if query:
        would_save: bool = query_user('Would you like to save the result?')

        if would_save and not auto:
            result_name: str = query_str('The name of result?').replace('\t ', '')

            result_path: str = global_config.RESULT_SAVE_PATH + '/' + get_class_name_str(model) + '/' + result_name

            describe_info: str = query_str('describe?')

            mkdir(result_path)

            with open(result_path + '/result.txt', 'w') as output:
                output.write('best {}: {}, {}\n'.format(eval_metric, best_performance, best_indexes))
                output.write(result_json)
                output.close()

            with open(result_path + '/config.txt', 'w') as output:
                output.write('rand seed: '+ str(random_seed) + '\n')
                output.write(model_config_json + '\n')
                output.write(train_config_json + '\n')
                output.write(evaluate_config_json + '\n')
                output.write(str(list(model.modules())) + '\n')
                output.close()

            with open(result_path + '/describe.txt', 'w') as output:
                output.write(describe_info + '\n')
                output.close()

        while True and not auto:
            command: int = query_int('Would you like to\n0. exit\n1. view result\n2. view config', {0, 1, 2})
            if command == 0:
                break
            elif command == 1:
                print(result_json)
            else:
                print(evaluate_config)
                print(train_config_json)
                print(model_config_json)

    return best_performance, best_indexes, fucking_result


if __name__ == '__main__':
    loader: ExplicitDataLoader = ExplicitDataLoader(
        dataset_path=global_config.DATASET_PATH + DATASET_PATH,
        device=DEVICE
    )

    best_metric_perform_list: list = []
    all_metric_results_list: list = []

    for seed in RANDOM_SEED_LIST:
        print()
        print('Begin seed:', seed)
        best_perform, _, all_metric_result = main(
            device=DEVICE, model_config=MODEL_CONFIG, train_config=TRAIN_CONFIG,
            evaluate_config=EVALUATE_CONFIG, data_loader=loader, random_seed=seed, query=False
        )

        best_metric_perform_list.append(best_perform)
        all_metric_results_list.append(all_metric_result)

    evaluate_config_str: str = json.dumps(EVALUATE_CONFIG, indent=4)
    model_config_str: str = json.dumps(MODEL_CONFIG, indent=4)
    train_config_str: str = json.dumps(TRAIN_CONFIG, indent=4)
    print(evaluate_config_str)
    print(model_config_str)
    print(train_config_str)

    merged_all_metric: dict = merge_dict(all_metric_results_list, _mean_merge_dict_func)

    merged_all_metric_str: str = json.dumps(merged_all_metric, indent=4)
    print(merged_all_metric_str)
    print('Best perform mean:', np.mean(best_metric_perform_list))
    print('Random seed list:', RANDOM_SEED_LIST)
