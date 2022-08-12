import datetime
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import pandas as pd


def mini_batch(batch_size: int, *tensors):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def progress_bar(now_index: int, total_num: int, start: float):
    total_num -= 1
    len = 20
    a = "*" * int((now_index + 1) / total_num * len)
    b = "." * int((max((total_num - now_index - 1), 0)) / total_num * len)
    c = (min(now_index + 1, total_num) / total_num) * 100
    dur = time.perf_counter() - start
    print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(c, a, b, dur), end="")


def random_color():
    color_arr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += color_arr[random.randint(0, 14)]
    return "#" + color


def draw_loss_pic(max_step: int, use_random_color: bool = False, **losses):

    plt.figure()
    for key in losses.keys():
        if use_random_color:
            plt.plot(range(1, max_step + 1), losses[key], color=random_color(), label=key)
        else:
            plt.plot(range(1, max_step + 1), losses[key], label=key)

    plt.legend()
    plt.show()


def draw_loss_pic_one_by_one(max_step: int, use_random_color: bool = False, **losses):

    for key in losses.keys():
        plt.figure()
        if use_random_color:
            plt.plot(range(1, max_step + 1), losses[key], color=random_color(), label=key)
        else:
            plt.plot(range(1, max_step + 1), losses[key], label=key)

        plt.legend()
        plt.show()


def draw_score_pic(x: list, use_random_color: bool = False, title: str = None, **losses):

    plt.figure()

    if title is not None:
        plt.title(title)

    for key in losses.keys():
        if use_random_color:
            plt.plot(x, losses[key], color=random_color(), label=key)
        else:
            plt.plot(x, losses[key],  label=key)

    plt.legend()
    plt.show()


def save_loss_pic(max_step: int, filename: str, use_random_color: bool = False, **losses):

    plt.figure()
    for key in losses.keys():
        if use_random_color:
            plt.plot(range(1, max_step + 1), losses[key], color=random_color(), label=key)
        else:
            plt.plot(range(1, max_step + 1), losses[key], label=key)
    plt.legend()
    plt.savefig(filename)


def save_loss_pic_one_by_one(max_step: int, dir_path: str, use_random_color: bool = False, **losses):
    for key in losses.keys():
        plt.figure()
        if use_random_color:
            plt.plot(range(1, max_step + 1), losses[key], color=random_color(), label=key)
        else:
            plt.plot(range(1, max_step + 1), losses[key], label=key)
        plt.legend()
        plt.savefig(dir_path + '/' + key + '.png')


def save_score_pic(x: list, filename: str, use_random_color: bool = False, **losses):

    plt.figure()
    for key in losses.keys():
        if use_random_color:
            plt.plot(x, losses[key], color=random_color(), label=key)
        else:
            plt.plot(x, losses[key],  label=key)
    plt.legend()
    plt.savefig(filename)


def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        return False


def get_now_time_str():
    now_time = datetime.datetime.now()
    time_str = now_time.strftime('%Y-%m-%d-%H-%M-%S')
    return time_str


def save_loss_list(filename: str, **losses):
    with open(filename, 'w') as output:
        for key in losses.keys():
            output.write(key + '\n')
            output.write(str(losses[key]))
            output.write('\n\n')
        output.close()


def build_paras_str(para_dict: dict) -> str:
    result: str = ''
    for key in para_dict.keys():
        result += (key + '=' + str(para_dict[key]) + '_')

    return result[: -1]


def merge_dict(dict_list: list, merge_func, **func_args):
    # assert len(dict_list) > 1, 'len(dict_list) should bigger than 1'
    first_dict: dict = dict_list[0]
    keys = first_dict.keys()
    for element_dict in dict_list:
        assert keys == element_dict.keys()

    result: dict = dict()
    for key in keys:
        elements_list: list = [element_dict[key] for element_dict in dict_list]
        result[key] = merge_func(elements_list, **func_args)

    return result


def _mean_merge_dict_func(elements_list, **args):
    # print(args)
    return np.mean(elements_list)


def _show_me_a_list_func(elements_list, **args):
    # print(args)
    return elements_list


def show_me_all_the_fucking_result(raw_result: dict, metric_list: list, k_list: list, best_index: int) -> dict:
    result_dict: dict = dict()
    for metric in metric_list:
        for k in k_list:
            temp_array: np.array = np.array(merge_dict(raw_result[metric], _show_me_a_list_func)[k])
            dict_key: str = str(metric) + '@' + str(k)
            result_dict[dict_key] = temp_array[best_index]
    return result_dict


def show_me_all_the_fucking_explicit_result(raw_result: dict, metric_list: list, best_index: int) -> dict:
    result_dict: dict = dict()
    for metric in metric_list:
        result_dict[metric] = raw_result[metric][best_index]
    return result_dict


def analyse_interaction_from_text(lines: list, has_value: bool = False):

    pairs: list = []

    users_set: set = set()
    items_set: set = set()

    for line in tqdm(lines):
        elements: list = line.split(',')
        user_id: int = int(elements[0])
        item_id: int = int(elements[1])
        if not has_value:
            pairs.append([user_id, item_id])
        else:
            value: float = float(elements[2])
            pairs.append([user_id, item_id, value])

        users_set.add(user_id)
        items_set.add(item_id)

    users_list: list = list(users_set)
    items_list: list = list(items_set)

    users_list.sort(reverse=False)
    items_list.sort(reverse=False)

    return pairs, users_list, items_list


def analyse_user_interacted_set(pairs: list):
    user_id_list: list = list()
    print('Init table...')
    for pair in tqdm(pairs):
        user_id, item_id = pair[0], pair[1]
        # user_bought_map.append(set())
        user_id_list.append(user_id)

    max_user_id: int = max(user_id_list)
    user_bought_map: list = [set() for i in range((max_user_id + 1))]
    print('Build mapping...')
    for pair in tqdm(pairs):
        user_id, item_id = pair[0], pair[1]
        user_bought_map[user_id].add(item_id)

    return user_bought_map


def transfer_loss_dict_to_line_str(loss_dict: dict) -> str:
    result_str: str = ''
    for key in loss_dict.keys():
        result_str += (str(key) + ': ' + str(loss_dict[key]) + ', ')

    result_str = result_str[0:len(result_str)-2]
    return result_str


def query_user(query_info: str) -> bool:
    print(query_info)
    while True:
        result = input('yes/no\n')
        if result in ['yes', 'no']:
            break
    return True if result == 'yes' else False


def query_str(query_info: str) -> str:
    result = input(query_info + '\n')
    return result


def query_int(query_info: str, int_range: set) -> int:
    print(query_info)
    while True:
        value = input('value range: ' + str(int_range) + '\n')
        try:
            result = int(value)
        except ValueError:
            continue
        if result not in int_range:
            continue
        return result


def get_class_name_str(obj) -> str:
    name: str = str(type(obj))

    l_index: int = name.index('\'')
    r_index: int = name.rindex('\'')
    return name[l_index + 1: r_index]
