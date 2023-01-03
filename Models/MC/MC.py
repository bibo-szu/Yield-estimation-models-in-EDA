"""

The implement of ground truth method: Monte-Carlo method.

"""

import numpy as np
import time
import os
from tool.util import write_data2csv, seed_set
import torch
import random

class MC():
    def __init__(self, f_norm, spice, initial_num=10000, sample_num=10, FOM_use_num=10, seed=0):
        self.x = None
        self.spice = spice
        self.f_norm = f_norm
        self.initial_num, self.sample_num, self.FOM_use_num = initial_num, sample_num, FOM_use_num
        self.seed = seed
        seed_set(seed)

    def save_result(self, P_fail, FOM, num, used_time, seed):
        data_info_list = [[P_fail], [FOM], [num], [used_time]]

        write_data2csv(tgt_dir=os.path.join("./results"),  # 保存目的文件
                       tgt_name=f"MC_case{self.spice.case}_{seed}.csv",  # 文件名:包含训练数据量, 模型名
                       head_info=('Pfail', 'FOM', 'num', 'used_time'),  # 表头
                       data_info=data_info_list)  # 信息

    def _get_new_y(self, sample_num, initial_num, i, test_y):
        if initial_num+(i+1)*sample_num > test_y.shape[0]:
            new_y = test_y[initial_num+i*sample_num:, :]
        else:
            new_y = test_y[initial_num+i*sample_num:initial_num+(i+1)*sample_num, :]
        return new_y

    def _evaluate(self, y, P_fail_list, FOM_use_num=10):
        indic = self.spice.indicator(y)
        P_fail = indic.sum() / indic.shape[0]
        P_fail_list = np.hstack([P_fail_list, P_fail])
        # leng = len(P_fail_list)

        if P_fail_list[-FOM_use_num:].mean() == 0:
            FOM = 1
        elif P_fail_list.shape[0] == 1:
            FOM = 1
        else:
            FOM = P_fail_list[-FOM_use_num:].std() / P_fail_list[-FOM_use_num:].mean()
        return P_fail, FOM

    def _get_initial_sample(self, initial_num, test_y):
        return test_y[0:initial_num,:]

    def rearrage_x(self, test_x, initial_num, sample_num):
        # np.random.shuffle(test_x)
        test_x = np.random.permutation(test_x)
        return test_x

    def start_estimate(self, max_num=10000000):

        initial_num, sample_num, FOM_use_num = self.initial_num, self.sample_num, self.FOM_use_num

        now_time = time.time()
        self.y, P_fail_list, FOM_list, data_num_list = np.empty([0,1]), np.empty([0]), np.empty([0]), np.empty([0])

        x = np.random.multivariate_normal(mean=np.zeros(self.spice.feature_num), cov=np.eye(self.spice.feature_num),
                                          size=initial_num)
        new_y = self.spice(x)
        self.y = np.vstack([self.y, new_y])
        P_fail, FOM = self._evaluate(self.y, P_fail_list, FOM_use_num)
        P_fail_list = np.hstack([P_fail_list, P_fail])

        self.save_result(P_fail=P_fail, FOM=FOM, num=self.y.shape[0], used_time=time.time() - now_time, seed=self.seed)
        print(f" # IS sample: {self.y.shape[0]}, fail_rate: {P_fail}, FOM: {FOM}")
        i=0

        while ((FOM>0.1) and (self.y.shape[0]<max_num)) or (i<10):

            x = self.f_norm.sample(n=sample_num)
            new_y = self.spice(x)

            self.y = np.vstack([self.y, new_y])

            P_fail, FOM = self._evaluate(self.y, P_fail_list, FOM_use_num)
            P_fail_list = np.hstack([P_fail_list, P_fail])

            self.save_result(P_fail=P_fail, FOM=FOM, num=self.y.shape[0], used_time=time.time() - now_time, seed=self.seed)
            print(f"[MC] # IS sample: {self.y.shape[0]}, fail_rate: {P_fail}, FOM: {FOM}")
            i += 1

        return P_fail_list, data_num_list


def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
