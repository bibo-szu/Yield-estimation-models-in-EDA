"""

Intro:
    Implement of ACS

paper:
    Shi X, Yan H, Wang J, et al. Adaptive clustering and sampling for high-dimensional and multi-failure-region SRAM yield analysis[C]//Proceedings of the 2019 International Symposium on Physical Design. 2019: 139-146.

"""

import os
import time

import numpy as np
from Models.ACS.multi_cone_cluster import cone_cluster
from Distribution.gmm_v2 import mixture_gaussian
import mpmath as mp
from tool.util import write_data2csv, seed_set

class ACS():
    def __init__(self, spice, f_norm, g_cal_val, initial_fail_num, initial_sample_each, IS_num, FOM_num, seed):
        """
            :param f_norm: the origin MC distribution, usually a Gaussian distribution
            :param initial_sample_each: IS sample number of each Gausian distribution in the GMM g(x)
            :param IS_num: the number of IS samples in each iteration round
            :param initial_fail_num: the number of initial failed sample
            :param initial_sample_each: the number of samples each time during initial sampling
            :param max_gen_times: the max number of IS iterations
            :param g_cal_val: the used variance of importance sampling distribution during yield calculation
            :param FOM_num: the used number of latest fail rates to calculate FOM
            :param seed: random seed
        """
        self.spice = spice
        self.seed = seed
        self.f_norm, self.g_cal_val, self.initial_fail_num, self.initial_sample_each, self.IS_num, \
        self.FOM_num = f_norm, g_cal_val, initial_fail_num, initial_sample_each, IS_num, FOM_num
        seed_set(seed)

    def _identify_fail(self, x, y, spice):
        """
            return the failed samples from given x,y only
        """
        label_fail = spice.indicator(y).reshape([-1])
        feat_num = spice.feature_num
        if label_fail.any():
            x_fail = x[label_fail, :]
            y_fail = y[label_fail, :].reshape([-1, 1])
        else:
            x_fail = np.empty([0,feat_num])
            y_fail = np.empty([0,1])
        return x_fail, y_fail

    def _construct_mixture_norm(self, weight, x_fail, g_val, labels):
        """
            construct the GMM according to weight of each failed sample x
            Note that: the labels is unused due to the fact that in the paper the ratio induced by the K-means can be divided out...
        """
        # get gmm model
        mix_model = mixture_gaussian(pi=weight, mu=x_fail, var_num=g_val)
        return mix_model

    def _calculate_weight(self, org_norm, x_fail,):
        """
            get weights of each normal distribution of the GMM g(x)
        """
        # get gmm ratio (pi)
        mp_exp_broad = np.frompyfunc(mp.exp, 1, 1)
        mp_log_broad = np.frompyfunc(mp.log, 1, 1)
        log_pdf_each = org_norm.log_pdf(x_fail)
        pdf_each = mp_exp_broad(log_pdf_each)
        pdf_sum = pdf_each.sum()
        weight = (pdf_each / pdf_sum).astype(np.double)
        return weight

    def _calculate_weight_Kmeans(self, org_norm, x_fail, labels, cluster_num):
        """
            get weights of each normal distribution of the GMM g(x)
        """
        # get gmm ratio (pi)
        mp_exp_broad = np.frompyfunc(mp.exp, 1, 1)
        mp_log_broad = np.frompyfunc(mp.log, 1, 1)
        log_pdf_each = org_norm.log_pdf(x_fail)
        pdf_each = mp_exp_broad(log_pdf_each)
        cluster_weight_sum_list = []

        weight = pdf_each

        for i in range(cluster_num):
            weight[labels==i] = weight[labels==i]/pdf_each[labels==i].sum()

        weight = weight/pdf_each.sum()

        for i in range(cluster_num):
            cluster_weight_sum = pdf_each[labels==i].sum()
            cluster_weight_sum_list.append(cluster_weight_sum)
            weight[labels==i] = cluster_weight_sum_list[i] * weight[labels==i]

        return weight.astype(np.double)


    def _calculate_val(self, x, y, f_x, g_x, spice):
        """
            calculate log f(x), log g(x) and I(x)
        """
        log_f_val = f_x.log_pdf(x).reshape([-1])
        log_g_val = g_x.log_pdf(x).reshape([-1])
        I_val = spice.indicator(y).reshape([-1])
        return log_f_val, log_g_val, I_val

    def _calculate_fail_rate(self, fail_rate_this_round, fail_rate_list):
        """
            calculate the overall P_f vias all IS samples
        """
        fail_rate = (sum(fail_rate_list) + fail_rate_this_round) / (len(fail_rate_list) + 1)
        return fail_rate

    def _calculate_fail_rate_this_round(self, log_f_val, log_g_val, I_val):
        """
            calculate P_f in a single IS round
        """
        IS_num = log_f_val.shape[0]
        w_val = np.exp(log_f_val - log_g_val)

        w_val[(w_val == np.inf)] = 1e290

        fail_rate_this_round = (w_val * I_val).sum() / IS_num
        return fail_rate_this_round

    def _calculate_FOM(self, fail_rate_list, FOM_num):
        """
            calculate FOM
        """
        length = len(fail_rate_list)
        assert length >= 1
        if length == 1 or np.mean(fail_rate_list[-FOM_num:]) == 0:
            return 1
        else:
            return np.std(fail_rate_list[-FOM_num:]) / np.mean(fail_rate_list[-FOM_num:])
    
    def _initial_sampling(self, initial_fail_num, sample_num_each_sphere, spice):
        """
            pre sampling using uniform distribution
        """
        captured_fail_data_num = 0
        iter_count = 0
        capture_any_fail_data_flag = False
        feat_num = spice.feature_num  # feature number of x

        while captured_fail_data_num < initial_fail_num:
            new_x = np.random.uniform(low=spice.low_bounds, high=spice.up_bounds, size=[sample_num_each_sphere, feat_num])
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
        sample_total_num = (iter_count+1) * sample_num_each_sphere

        return x_samples, sample_total_num

    def _save_result(self, P_fail, FOM, num, used_time, seed):
        data_info_list = [[P_fail], [FOM], [num], [used_time]]

        write_data2csv(tgt_dir=os.path.join("./results"),  # 保存目的文件
                       tgt_name=f"ACS_case{self.spice.case}_{seed}.csv",  # 文件名:包含训练数据量, 模型名
                       head_info=('Pfail', 'FOM', 'num', 'used_time'),  # 表头
                       data_info=data_info_list)  # 信息

    def start_estimate(self, max_num=100000):
        """
            call this function to start the yield estimation process,
            and the numerical results will be saved in "./results/HSCS_case*.csv" automatically.
        """

        f_norm, g_cal_val, initial_fail_num, initial_sample_each, IS_num, FOM_num = self.f_norm, self.g_cal_val, self.initial_fail_num, self.initial_sample_each, self.IS_num, self.FOM_num

        time1 = time.time()

        self.x_fail, origin_sample_num = self._initial_sampling(initial_fail_num, initial_sample_each, self.spice)
        self.y_fail = self.spice(self.x_fail)

        k = round(np.sqrt(initial_fail_num))
        d = self.x_fail.shape[-1]
        classifier = cone_cluster(cluster_num=k, dim=self.x_fail.shape[-1]) # K-means classifier

        fail_rate_list = []
        FOM_list = []
        fail_rate_this_round_list = []
        iter_count = 0
        FOM=1

        while ((FOM>=0.1)and(iter_count*IS_num+origin_sample_num<max_num)) or (iter_count<10):
            self.label_fail = classifier.cluster(self.x_fail)  # not used

            # weights in the gmm of each normal distribution
            weight_list = self._calculate_weight_Kmeans(org_norm=f_norm, x_fail=self.x_fail, labels=self.label_fail, cluster_num=k)

            # the g'(x) used for sampling
            mix_gaussian_val = self._construct_mixture_norm(weight_list, self.x_fail, g_cal_val, self.label_fail)

            # IS sampling
            x_IS = mix_gaussian_val.sample(n=IS_num)
            y_IS = self.spice(x_IS)

            # collect only the failed samples from the IS samples
            new_x_fail, new_y_fail = self._identify_fail(x=x_IS, y=y_IS, spice=self.spice)
            self.x_fail = np.vstack([self.x_fail, new_x_fail])
            self.y_fail = np.vstack([self.y_fail, new_y_fail])

            # get log f(x), log g(x) and I(x) of IS samples
            log_f_IS_val, log_g_IS_val, I_IS_val = self._calculate_val(x_IS, y_IS, f_norm, mix_gaussian_val, self.spice)

            # the fail_rate calculated only using IS samples of this iteration round
            fail_rate_this_round = self._calculate_fail_rate_this_round(log_f_IS_val, log_g_IS_val, I_IS_val)

            # the real overall fail_rate after this iteration
            fail_rate = self._calculate_fail_rate(fail_rate_this_round, fail_rate_this_round_list)

            fail_rate_this_round_list.append(fail_rate_this_round)
            fail_rate_list.append(fail_rate)

            # calculate FOM
            FOM = self._calculate_FOM(fail_rate_list, FOM_num)
            FOM_list.append(FOM)

            iter_count+=1

            self._save_result(fail_rate, FOM, iter_count*IS_num+origin_sample_num, time.time()-time1, self.seed)

            print(f"num:{iter_count*IS_num+origin_sample_num}, pfail:{fail_rate}, FOM:{FOM}")

