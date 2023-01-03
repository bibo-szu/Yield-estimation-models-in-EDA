"""
Intro:
    Implement of MNIS

paper:
    Dolecek L, Qazi M, Shah D, et al. Breaking the simulation barrier: SRAM evaluation through norm minimization[C]//2008 IEEE/ACM International Conference on Computer-Aided Design. IEEE, 2008: 322-329.

"""

import numpy as np
from Distribution.normal_v1 import norm_dist
import time
import os
from tool.util import write_data2csv, seed_set

class MNIS():
    def __init__(self, spice, f_norm, g_sam_val, initial_fail_num, initial_sample_each, IS_num, FOM_num, seed=0):
        """
            :param f_norm: the origin MC distribution, usually a Gaussian distribution
            :param initial_fail_num: the number of initial failed sample
            :param initial_sample_each: the number of samples each time during initial sampling
            :param IS_num: the number of importance sampling (IS) during each importance sampling iteration
            :param g_sam_val: the used variance of importance distribution during sampling
            :param FOM_num: the used number of latest fail rates to calculate FOM
            :param seed: ran dom seed
        """

        self.spice = spice
        self.f_norm, self.g_sam_val, self.initial_fail_num, self.initial_sample_each, self.IS_num, self.FOM_num = f_norm, g_sam_val, initial_fail_num, initial_sample_each, IS_num, FOM_num
        self.seed = seed
        seed_set(seed)

    def _calculate_val(self, x, y, f_x, g_x, spice):
        log_f_val = f_x.log_pdf(x).reshape([-1])
        log_g_val = g_x.log_pdf(x).reshape([-1])
        I_val = spice.indicator(y).reshape([-1])
        return log_f_val, log_g_val, I_val

    def _calculate_fail_rate_this_round(self, log_f_val, log_g_val, I_val):
        IS_num = log_f_val.shape[0]
        w_val = np.exp(log_f_val - log_g_val)

        w_val[(w_val == np.inf)] = 1e290

        fail_rate_this_round = (w_val * I_val).sum() / IS_num
        return fail_rate_this_round

    def _calculate_fail_rate(self, fail_rate_this_round, fail_rate_list):
        fail_rate = (sum(fail_rate_list) + fail_rate_this_round) / (len(fail_rate_list) + 1)
        return fail_rate

    def _calculate_FOM(self, fail_rate_list, FOM_num):
        length = len(fail_rate_list)
        assert length >= 1
        if length == 1 or np.mean(fail_rate_list[-FOM_num:]) == 0:
            return 1
        else:
            # print("std", np.std(fail_rate_list[-FOM_num:]))
            return np.std(fail_rate_list[-FOM_num:]) / np.mean(fail_rate_list[-FOM_num:])

    def _save_result(self, P_fail, FOM, num, used_time, seed):
        data_info_list = [[P_fail], [FOM], [num], [used_time]]

        write_data2csv(tgt_dir=os.path.join("./results"),  # 保存目的文件
                       tgt_name=f"MNIS_case{self.spice.case}_{seed}.csv",  # 文件名:包含训练数据量, 模型名
                       head_info=('Pfail', 'FOM', 'num', 'used_time'),  # 表头
                       data_info=data_info_list)  # 信息

    def _initial_sampling(self, initial_fail_num, sample_num_each_sphere, spice):
        captured_fail_data_num = 0
        iter_count = 0
        capture_any_fail_data_flag = False
        feat_num = spice.feature_num  # feature number of x

        while captured_fail_data_num < initial_fail_num:
            new_x = np.random.uniform(low=spice.low_bounds, high=spice.up_bounds,
                                      size=[sample_num_each_sphere, feat_num])
            new_y = spice(new_x)

            y_labels = spice.indicator(new_y).reshape([-1])

            if y_labels.any() == True:
                failed_x = new_x[y_labels]

                if capture_any_fail_data_flag == False:
                    x_samples = failed_x
                else:
                    x_samples = np.vstack([x_samples, failed_x])

                capture_any_fail_data_flag = True
                captured_fail_data_num += failed_x.shape[0]

            iter_count += 1
            print(iter_count, y_labels.any())

        x_samples = x_samples[0:initial_fail_num, :]  # discard excess samples
        sample_total_num = (iter_count + 1) * sample_num_each_sphere
        return x_samples, sample_total_num

    def start_estimate(self, max_num=None):
        """
            call this function to start the yield estimation process,
            and the numerical results will be saved in "./results/MNIS_case3.csv" automatically.
        """

        f_norm, g_sam_val, initial_fail_num, initial_sample_each, IS_num, FOM_num = self.f_norm, self.g_sam_val, self.initial_fail_num, self.initial_sample_each, self.IS_num, self.FOM_num

        time1 = time.time()

        self.x_fail, origin_sample_num = self._initial_sampling(initial_fail_num, initial_sample_each, self.spice)
        self.y_fail = self.spice(self.x_fail)

        d = self.x_fail.shape[-1]

        min_norm = self.x_fail[abs(self.x_fail).sum(-1).argmin(), :]

        g_sam_norm = norm_dist(mu=min_norm, var=np.eye(d)*g_sam_val)

        fail_rate_list = []
        FOM_list = []
        fail_rate_this_round_list = []

        iter_count = 0

        if max_num == None:
            max_reach_flag = True
        else:
            max_reach_flag = iter_count * IS_num + origin_sample_num < max_num
        FOM = 1

        while ((max_reach_flag) and (FOM>0.1)) or (iter_count<10):
            self.label_fail = None

            # IS samples
            x_IS = g_sam_norm.sample(n=IS_num)
            y_IS = self.spice(x_IS)

            # get log f(x), log g(x) and I(x)
            log_f_IS_val, log_g_IS_val, I_IS_val = self._calculate_val(x_IS, y_IS, f_norm, g_sam_norm, self.spice)

            # the fail_rate calculated only using IS samples of this iteration round
            fail_rate_this_round = self._calculate_fail_rate_this_round(log_f_IS_val, log_g_IS_val, I_IS_val)

            # the real overall fail_rate after this iteration
            fail_rate = self._calculate_fail_rate(fail_rate_this_round, fail_rate_this_round_list)

            fail_rate_this_round_list.append(fail_rate_this_round)
            fail_rate_list.append(fail_rate)

            FOM = self._calculate_FOM(fail_rate_list, FOM_num)
            FOM_list.append(FOM)

            iter_count += 1

            self._save_result(fail_rate, FOM, iter_count*IS_num+origin_sample_num, time.time()-time1, self.seed)
            print(f"[MNIS] num:{iter_count * IS_num + origin_sample_num}, pfail:{fail_rate}, FOM:{FOM}")
            max_reach_flag = iter_count * IS_num + origin_sample_num < max_num


