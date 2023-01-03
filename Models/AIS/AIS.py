"""

Intro:
    Implement of AIS

paper:
    Shi X, Liu F, Yang J, et al. A fast and robust failure analysis of memory circuits using adaptive importance sampling method[C]//2018 55th ACM/ESDA/IEEE Design Automation Conference (DAC). IEEE, 2018: 1-6.

"""

from sampyl import np
from Distribution.normal_v1 import norm_dist
from Distribution.gmm_v2 import mixture_gaussian
import torch.nn as nn
import os
import time
from tool.util import write_data2csv, seed_set


class AIS(nn.Module):

    def __init__(self, spice, f_norm, g_cal_num=1, origin_sam_bound_num=1, initial_failed_data_num=50,
                       num_generate_each_norm=1, sample_num_each_sphere=1000, max_gen_times=10,
                       FOM_num=10, seed=0):
        """
            :param f_norm: the origin MC distribution, usually a Gaussian distribution
            :param num_generate_each_norm: IS sample number of each Gausian distribution in the GMM g(x)
            :param origin_sam_bound_num: the zooming scale factor of bound of initial sampling area
            :param initial_failed_data_num: the number of initial failed sample
            :param sample_num_each_sphere: the number of samples each time during initial sampling
            :param max_gen_times: the max number of IS iterations
            :param FOM_num: the used number of latest fail rates to calculate FOM
            :param seed: random seed
        """

        super(AIS, self).__init__()
        self.spice = spice
        self.low_bounds = spice.low_bounds
        self.up_bounds = spice.up_bounds

        self.x_samples = None
        self.y_samples = None

        self.f_norm, self.g_cal_num, self.origin_sam_bound_num, self.initial_failed_data_num, \
        self.num_generate_each_norm, self.sample_num_each_sphere, self.max_gen_times, \
        self.FOM_num = f_norm, g_cal_num, origin_sam_bound_num, initial_failed_data_num, \
        num_generate_each_norm, sample_num_each_sphere, max_gen_times, FOM_num

        self.seed = seed
        seed_set(seed)

    def save_result(self, P_fail, FOM, num, used_time, seed):
            data_info_list = [[P_fail], [FOM], [num], [used_time]]
            write_data2csv(tgt_dir=os.path.join("./results"),  # 保存目的文件
                           tgt_name=f"AIS_case{self.spice.case}_{seed}.csv",  # 文件名:包含训练数据量, 模型名
                           head_info=('Pfail', 'FOM', 'num', 'used_time'),  # 表头
                           data_info=data_info_list)  # 信息

    def indicator_func(self,y):
        """
            I(X): if the corresponding  y of sample x is failed, return True, otherwise False.
        """
        return y > self.spice.threshold


    def pre_sampling(self, initial_failed_data_num, sample_num_each_sphere, radius_interval=0.3, origin_sam_bound_num=1):
        """
            initial sampling
        """
        feat_num = self.spice.feature_num  # feature number of x

        captured_fail_data_num = 0
        iter_count = 0
        capture_any_fail_data_flag = False

        x_samples = None

        while captured_fail_data_num < initial_failed_data_num:

            # define sphere radius
            max_radius = self.up_bounds.min()
            radius = ((iter_count+1)*radius_interval) - (((iter_count+1)*radius_interval)//max_radius)*max_radius

            # capture failed samples
            # new_x = sample_sphere(num=sample_num_each_sphere,dim=feat_num,radius=radius)
            new_x = np.random.uniform(low=origin_sam_bound_num * self.spice.low_bounds, high=origin_sam_bound_num * self.spice.up_bounds, size=[sample_num_each_sphere,feat_num])

            new_y = self.spice(new_x)
            y_labels = self.indicator_func(new_y).reshape([-1])

            if y_labels.any() == True:
                failed_x = new_x[y_labels]

                if capture_any_fail_data_flag == False:
                    x_samples = failed_x
                else:
                    x_samples = np.vstack([x_samples,failed_x])

                capture_any_fail_data_flag = True
                captured_fail_data_num += failed_x.shape[0]

            iter_count += 1
            print(iter_count, radius, y_labels.any())

        x_samples = x_samples[0:initial_failed_data_num, :]  # discard excessive samples
        initial_sample_total_num = iter_count * sample_num_each_sphere

        return x_samples, initial_sample_total_num

    def calculate_weights(self, log_f_val, log_g_val, I_val):
        """
            calculate weights
        """
        weight_list = np.exp(log_f_val - log_g_val) * I_val
        return weight_list

    def _calculate_fail_rate(self, weight_sum_list, N):
        """
            calculate the overall P_f
        """
        fail_rate = sum(weight_sum_list) / N / len(weight_sum_list)
        return fail_rate

    def _calculate_FOM(self, fail_rate_list, FOM_num):
        length = len(fail_rate_list)
        assert length >= 1
        if length == 1 or np.mean(fail_rate_list[-FOM_num:]) == 0:
            return 1
        else:
            return np.std(fail_rate_list[-FOM_num:]) / np.mean(fail_rate_list[-FOM_num:])

    def resampling(self, x, y, weight_list, data_num):
        """
            resamples according to weight_list
        """
        if sum(weight_list) != 0:
            weight_list = weight_list / sum(weight_list)
            weight_list = weight_list.astype(np.float64)

            # 均匀采样
            new_weight_list = np.round(weight_list * data_num).astype(int)
            sample_index = []
            for idx, i in enumerate(new_weight_list):
                for j in range(i):
                    sample_index.append(idx)
            sample_index = np.array(sample_index)
            x_samples = x[sample_index,:]
            y_samples = y[sample_index,:]

        else:
            x_samples = x
            y_samples = y

        return x_samples, y_samples

    def _construct_mixture_norm(self, minnorm_point, betas, spice, g_var_num):
        """
            the labels is unused due to the fact that in the paper the ratio induced by the K-means can be divided out...
        """
        feat_num = spice.feature_num
        mean = minnorm_point
        pi = betas
        mix_model = mixture_gaussian(pi=pi, mu=mean, var_num=g_var_num)
        return mix_model

    def _calculate_val(self, x, y, f_x, g_x, spice):
        """
            calculate log f(x), log g(x) and I(x)
        """
        log_f_val = f_x.log_pdf(x).reshape([-1])
        log_g_val = g_x.log_pdf(x).reshape([-1])
        I_val = spice.indicator(y).reshape([-1])
        return log_f_val, log_g_val, I_val


    def start_estimate(self, max_num=1000000):
        """
            call this function to start the yield estimation process,
            and the numerical results will be saved in "./results/ACS_case*.csv" automatically.
        """

        f_norm, g_cal_num, origin_sam_bound_num, initial_failed_data_num, \
        num_generate_each_norm, sample_num_each_sphere, max_gen_times, FOM_num =  self.f_norm,\
        self.g_cal_num, self.origin_sam_bound_num, self.initial_failed_data_num, \
        self.num_generate_each_norm, self.sample_num_each_sphere, self.max_gen_times, self.FOM_num

        # get initial failure data
        self.x_samples, initial_sample_total_num = self.pre_sampling(initial_failed_data_num, sample_num_each_sphere, origin_sam_bound_num=origin_sam_bound_num)
        self.y_samples = self.spice(self.x_samples)

        print(f"# capture {initial_failed_data_num} initial failed data in {initial_sample_total_num} sampling")

        # initialize variable
        fail_rate_list = []
        FOM_list = []
        data_num_list = []
        weight_sum_list = []
        FOM_metric = np.inf
        gen_time = 0
        gen_data_num = initial_failed_data_num  # number of data to generate, equal to initial failed data num
        now_time = time.time()

        sample_data_num = 0
        # start iteration, till fail_rate converges or simulation times reach the pre-defined maximum
        while (sample_data_num<max_num) and (FOM_metric >= 0.1) or (gen_time<10):
            gen_time += 1 # IS times
            x_fail_num = self.x_samples.shape[0]

            # g(x) for IS sampling
            norm_mixture = self._construct_mixture_norm(minnorm_point=self.x_samples, betas=np.ones([x_fail_num])/x_fail_num,
                                                         spice=self.spice, g_var_num=g_cal_num)

            # generate new samples from g(x)
            IS_x_samples = norm_mixture.sample(n=num_generate_each_norm * initial_failed_data_num)
            IS_y_samples = self.spice(IS_x_samples)

            # calculate log f(x), log g(x) and I(x) of IS samples
            log_f_val, log_g_val, I_val = self._calculate_val(IS_x_samples, IS_y_samples, f_norm, norm_mixture, self.spice)
            # calculate the weight of each x
            weight_list = self.calculate_weights(log_f_val, log_g_val, I_val)

            weight_sum_list.append(sum(weight_list))

            # calculate the overall P_f
            fail_rate = self._calculate_fail_rate(weight_sum_list, N=gen_data_num * num_generate_each_norm)
            fail_rate_list.append(fail_rate)

            # calculate FOM
            FOM_metric = self._calculate_FOM(fail_rate_list, FOM_num)
            FOM_list.append(FOM_metric)

            # calculate weights of the mean vector in last g(x)
            log_f_val_origin, log_g_val_origin, I_val_origin = self._calculate_val(self.x_samples, self.y_samples, f_norm, norm_mixture, self.spice)
            origin_weight_list = self.calculate_weights(log_f_val_origin, log_g_val_origin, I_val_origin)

            sample_data_num = initial_sample_total_num + gen_time * gen_data_num * num_generate_each_norm

            # resample failed samples from the new IS failed samples and origin failed samples according to their weights
            self.x_samples, self.y_samples = self.resampling(x=np.vstack([self.x_samples, IS_x_samples]),
                                                             y=np.vstack([self.y_samples, IS_y_samples]),
                                                             weight_list=np.hstack([origin_weight_list, weight_list]),
                                                             data_num=gen_data_num)

            data_num_list.append(sample_data_num)

            # save metric & data sampled number
            self.save_result(fail_rate, FOM_metric, num=sample_data_num, used_time=time.time() - now_time, seed=self.seed)
            print(f"[AIS] # already sample {sample_data_num} data, fail_rate: {fail_rate}, FOM: {FOM_metric}")

        return fail_rate_list, FOM_list, data_num_list

