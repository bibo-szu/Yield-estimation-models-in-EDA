"""
Intro:
    Implement of HSCS

paper:
    Wu W, Bodapati S, He L. Hyperspherical clustering and sampling for rare event analysis with multiple failure region coverage[C]//Proceedings of the 2016 on International Symposium on Physical Design. 2016: 153-160.

"""

from sampyl import np
import torch.nn as nn
from sklearn.cluster import KMeans
import os
import time

from Distribution.gmm_v2 import mixture_gaussian
from tool.util import write_data2csv, seed_set

class HSCS(nn.Module):

    def __init__(self, spice, f_norm, g_var_num=1, bound_num=1, IS_sample_num=100,
                       initial_failed_data_num=50, sample_num_each_sphere=1000,
                       max_gen_times=10, ratio=0.1, FOM_num=10, find_MN_sam_num=100, seed=0):
        """
            :param f_norm: the origin MC distribution, usually a Gaussian distribution
            :param bound_num: the zooming scale factor of bound of initial sampling area
            :param find_MN_sam_num: number using find MN samples
            :param IS_sample_num: the number of IS samples in each iteration round
            :param initial_failed_data_num: the number of initial failed sample
            :param sample_num_each_sphere: the number of samples each time during initial sampling
            :param IS_sample_num: the number of importance sampling (IS) during each importance sampling iteration
            :param max_gen_times: the max number of IS iterations
            :param g_var_num: the used variance of importance sampling distribution during yield calculation
            :param ratio: how large the ratio of f(x) mixing in g(x) is
            :param FOM_num: the used number of latest fail rates to calculate FOM
        """
        super(HSCS, self).__init__()
        self.spice = spice
        self.low_bounds = spice.low_bounds
        self.up_bounds = spice.up_bounds

        self.x_samples = None
        self.y_samples = None

        self.seed = seed
        self.f_norm, self.g_var_num, self.bound_num, self.IS_sample_num, self.initial_failed_data_num, \
        self.sample_num_each_sphere, self.max_gen_times, self.ratio, self.FOM_num,\
        self.find_MN_sam_num = f_norm, g_var_num, bound_num, IS_sample_num, initial_failed_data_num,\
                               sample_num_each_sphere, max_gen_times, ratio, FOM_num, find_MN_sam_num
        seed_set(seed)


    def indicator_func(self,y):
        """
        I(X): if the corresponding  y of sample x is failed, return True, otherwise False.
        """
        return y > self.spice.threshold

    def sample_on_sphere(self, num, dim, radius, direction = None):
        if type(direction) == type(None):  # No direction defined, then sample on the whole sphere
            x = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), num)
            x = self.normr(x) * (radius)
            return x

        else:  # sample on sphere with particular R and dirction
            centroid = radius * direction
            total_bounds = self.spice.up_bounds - self.spice.low_bounds
            up_bound = centroid + total_bounds * 0.1
            low_bound = centroid - total_bounds * 0.1

            # to make sure samples drawn within bounds constrain
            if (up_bound > self.spice.up_bounds).any() == True:
                up_bound[(up_bound > self.spice.up_bounds)] = self.spice.up_bounds[(up_bound > self.spice.up_bounds).reshape([-1])] # reshape -1 to handle the size
            if (low_bound < self.spice.low_bounds).any() == True:
                low_bound[(low_bound < self.spice.low_bounds)] = self.spice.low_bounds[(low_bound < self.spice.low_bounds).reshape([-1])]
            x = np.random.uniform(low=low_bound, high=up_bound, size=[num, self.spice.feature_num])

            return x

    def pre_sampling(self, initial_failed_data_num, sample_num_each_sphere, bound_num, radius_interval=0.3):
        """
            initial sampling using uniform distribution
        """
        feat_num = self.spice.feature_num  # feature number of x
        pre_sampling_num_list = []

        captured_fail_data_num = 0
        iter_count = 0
        capture_any_fail_data_flag = False

        x_samples = None
        failed_x = None

        while captured_fail_data_num < initial_failed_data_num:

            # define sphere radius
            max_radius = self.up_bounds.min()
            radius = ((iter_count+1)*radius_interval) - (((iter_count+1)*radius_interval)//max_radius)*max_radius

            # capture failed samples
            # new_x = sample_sphere(num=sample_num_each_sphere,dim=feat_num,radius=radius) #
            # new_x = self.sample_on_sphere(num=sample_num_each_sphere, dim=feat_num, radius=radius)
            new_x = np.random.uniform(low=bound_num * self.spice.low_bounds, high=bound_num * self.spice.up_bounds, size=[sample_num_each_sphere,feat_num])

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

            pre_sampling_num_list.append(iter_count * sample_num_each_sphere)

        x_samples = x_samples[0:initial_failed_data_num, :]  # discard excess samples
        # initial_sample_total_num = iter_count * sample_num_each_sphere

        return x_samples, pre_sampling_num_list

    def normr(self, x, verticalize=True):
        """
            return the normalized unit direction of sample x
        """
        if verticalize:
            return x / np.sqrt((x*x).sum(axis=-1)).reshape([-1,1])
        else:
            return x / np.sqrt((x * x).sum(axis=-1))

    def get_radius(self, x):
        """
            return the radius of sample x
        """
        return np.sqrt((x * x).sum(axis=-1)).reshape([-1,1])

    def get_weights(self, x):
        """
            get the weights
        """
        return np.exp(-self.get_radius(x)) / np.exp(-self.get_radius(x).max())

    def weighted_sphere_Kmeans(self, x_samples, k, weights, multi_start_num=5):
        """
            cluster x_samples and return their centroids

            :param x_samples: failed data
            :param k: cluster_num
            :param weights: the weights of x_samples

            :return x_labels: labels of x
            :return now_centroids: centroids of each clusters
            :return new_k: new cluster number
        """
        i = 1
        new_k = k
        now_centroids = KMeans(n_clusters=k, random_state=i).fit(x_samples).cluster_centers_
        now_centroids = self.normr(now_centroids, verticalize=True)
        if_centroids_stable = False

        while(if_centroids_stable == False):
            last_centroids = now_centroids
            # classify samples
            argmax_matrix = np.dot(x_samples, now_centroids.T) # TODO: 这里的centroid好像没有归一化? fixed
            x_labels = argmax_matrix.argmax(axis=-1)

            # update k and remove empty cluster
            k_temp = 0
            x_labels_copy = x_labels.copy()
            for j in range(new_k):
                if (x_labels_copy == j).any():
                    x_labels[x_labels_copy == j] = k_temp
                    k_temp += 1
            new_k = k_temp

            # update centroid
            now_centroids_list = []
            for j in range(new_k):
                x_j = x_samples[(x_labels==j), :].copy()
                weight_j = weights[(x_labels==j), :].copy()
                mu = (x_j * weight_j).sum(axis=0)
                mu = self.normr(mu, verticalize=False)
                now_centroids_list.append(mu)
            now_centroids = np.array(now_centroids_list)

            # if centroids remain unchanged, stop iteration
            if now_centroids.shape == last_centroids.shape:
                if_centroids_stable = (now_centroids == last_centroids).all()

        return x_labels, now_centroids, new_k

    def find_min_norm(self, x_samples, centroids, cluster_num, find_MN_sam_num=100):
        """
            return the min_norm point of each cluster
        """
        spend_num = 0
        dim = self.spice.feature_num
        min_norm_points = []


        for j in range(cluster_num): #  k cluster

            R_max = self.get_radius(x_samples).min()
            R_min = 0
            R_thre = (R_max - R_min)*0.05 # threshold
            direction = self.normr(centroids[j, :])

            # start iteration to find min-norm point R
            while (R_max - R_min) >= R_thre:
                spend_num += find_MN_sam_num
                R = (R_max + R_min) / 2
                samples = self.sample_on_sphere(num=find_MN_sam_num, dim=dim, radius=R, direction=direction)
                samples_y = self.spice(samples)  # TODO: the bug1 fixed
                if self.indicator_func(samples_y).any():
                    R_max = R
                else:
                    R_min = R

            # save min-norm point R in this cluster
            min_norm_point = R * self.normr(centroids[j, :], verticalize=False)
            min_norm_points.append(min_norm_point)

        # turn list into array
        min_norm_points = np.array(min_norm_points)
        return min_norm_points, spend_num

    def get_cluster_ratios(self, weights, x_labels, k): # beta
        """
            return the ratio of mixture of each cluster in g(x)
        """
        cluster_ratios = []
        for cluster in range(k):
            beta = weights[(x_labels==cluster),:].sum() / weights.sum()
            cluster_ratios.append(beta)
        cluster_ratios = np.array(cluster_ratios)
        return cluster_ratios

    def _construct_mixture_norm(self, minnorm_point, betas, ratios, spice, g_var_num):
        """
            the labels is unused due to the fact that in the paper the ratio induced by the K-means can be divided out...
        """
        feat_num = spice.feature_num
        mean = np.vstack([np.zeros([1,feat_num]), minnorm_point])
        pi = np.hstack([ratios, (1-ratios) * betas ])
        mix_model = mixture_gaussian(pi=pi, mu=mean, var_num=g_var_num)
        return mix_model

    def save_result(self, P_fail, FOM, num, used_time, seed):
        data_info_list = [[P_fail], [FOM], [num], [used_time]]
        write_data2csv(tgt_dir=os.path.join("results"),  # 保存目的文件
                       tgt_name=f"HSCS_case{self.spice.case}_{seed}.csv",  # 文件名:包含训练数据量, 模型名
                       head_info=('Pfail', 'FOM', 'num', 'used_time'),  # 表头
                       data_info=data_info_list)  # 信息

    def _calculate_fail_rate_this_round(self, log_f_val, log_g_val, I_val):
        """
            return the fail rate using only samples of this iteration round
            :param log_f_val: log f(x)
            :param log_g_val: log g(x)
            :param I_val:     I(x)
        """
        IS_num = log_f_val.shape[0]
        w_val = np.exp(log_f_val - log_g_val)

        w_val[(w_val == np.inf)] = 1e290

        fail_rate_this_round = (w_val * I_val).sum() / IS_num
        return fail_rate_this_round

    def _calculate_fail_rate(self, fail_rate_this_round, fail_rate_list):
        """
            return the fail rate using all IS samples
        """
        fail_rate = (sum(fail_rate_list) + fail_rate_this_round) / (len(fail_rate_list) + 1)
        return fail_rate

    def _calculate_FOM(self, fail_rate_list, FOM_num):
        length = len(fail_rate_list)
        assert length >= 1
        if length == 1 or np.mean(fail_rate_list[-FOM_num:]) == 0:
            return 1
        else:
            return np.std(fail_rate_list[-FOM_num:]) / np.mean(fail_rate_list[-FOM_num:])

    def _calculate_val(self, x, y, f_x, g_x, spice):
        """
            calculate log f(x), log g(x) and I(x)
        """
        log_f_val = f_x.log_pdf(x).reshape([-1])
        log_g_val = g_x.log_pdf(x).reshape([-1])
        I_val = spice.indicator(y).reshape([-1])
        return log_f_val, log_g_val, I_val

    def start_estimate(self, max_num=10000):
        """
            call this function to start the yield estimation process,
            and the numerical results will be saved in "./results/HSCS_case*.csv" automatically.
        """
        f_norm, g_var_num, bound_num, IS_sample_num, initial_failed_data_num, \
        sample_num_each_sphere, max_gen_times, ratio, FOM_num, find_MN_sam_num = self.f_norm, self.g_var_num,\
        self.bound_num, self.IS_sample_num, self.initial_failed_data_num, \
        self.sample_num_each_sphere, self.max_gen_times, self.ratio, self.FOM_num, self.find_MN_sam_num

        now_time = time.time()

        # get initial failed samples
        self.x_samples, pre_sampling_num_list = self.pre_sampling(initial_failed_data_num, sample_num_each_sphere, bound_num)
        self.y_samples = self.spice(self.x_samples)
        self.weights = self.get_weights(self.x_samples)

        # run WS K-means
        self.k = round(np.sqrt(self.x_samples.shape[0]))
        print(f"# k: {self.k}") # cluster num
        self.x_labels, self.centroids, self.k = self.weighted_sphere_Kmeans(self.x_samples, self.k, self.weights)

        # get min norm points
        min_norm_points, spend_num = self.find_min_norm(self.x_samples, self.centroids, self.k, find_MN_sam_num=find_MN_sam_num)

        ratio = ratio
        betas = self.get_cluster_ratios(self.weights, self.x_labels, self.k)  # get the ratios of each cluster

        # the GMM g(x)
        mix_model_val = self._construct_mixture_norm(minnorm_point=min_norm_points, betas=betas, ratios=ratio,
                                                     spice=self.spice, g_var_num=g_var_num)

        FOM = np.inf
        IS_time = 0
        fail_rate_list = []
        FOM_list = []
        fail_rate_this_round_list = []
        total_num = 0
        # start iteration, till fail_rate converges, or Importance sampling times reach pre-defined maximum
        while ((FOM >= 0.1) and (total_num<max_num)) or (IS_time<10):
            IS_time += 1  # IS times
            x_IS = mix_model_val.sample(n=IS_sample_num)  # sample from the GMM g(x)

            y_IS = self.spice(x_IS)

            log_f_IS_val, log_g_IS_val, I_IS_val = self._calculate_val(x_IS, y_IS, f_norm, mix_model_val, self.spice)

            # P_f using IS samples of this round
            fail_rate_this_round = self._calculate_fail_rate_this_round(log_f_IS_val, log_g_IS_val, I_IS_val)

            # the real P_f using all IS samples
            fail_rate = self._calculate_fail_rate(fail_rate_this_round, fail_rate_this_round_list)

            fail_rate_this_round_list.append(fail_rate_this_round)
            fail_rate_list.append(fail_rate)

            # calculate FOM
            FOM = self._calculate_FOM(fail_rate_list, FOM_num)
            FOM_list.append(FOM)

            total_num = pre_sampling_num_list[-1] + IS_time * y_IS.shape[0] + spend_num

            self.save_result(fail_rate, FOM, num=total_num, used_time=time.time() - now_time, seed=self.seed) # save data
            print(f" # Total sample: {total_num}, fail_rate: {fail_rate}, FOM: {FOM}")
