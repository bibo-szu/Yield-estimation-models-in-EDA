"""
    To start the process of yield estimation, run this script.
"""

import numpy as np
from tool.script_util import show_result, run_models
from SPICE.SPICE_case2 import SPICE_Case2
from Distribution.normal_v1 import norm_dist
from Models.MC.MC import MC
from Models.MNIS.MNIS import MNIS
from Models.AIS.AIS import AIS
from Models.HSCS.HSCS import HSCS
from Models.ACS.ACS import ACS


if __name__ == "__main__":
    # set the spice and the failure threshold
    spice = SPICE_Case2()
    spice.threshold = 5.3

    # set f(x)
    f_norm = norm_dist(mu=np.zeros(spice.feature_num), var=np.eye(spice.feature_num)*1)

    # yield estimation models to run
    model_list = [
                  MC(f_norm=f_norm, spice=spice, initial_num=1000, sample_num=10000, FOM_use_num=100, seed=0),
                  AIS(spice=spice, f_norm=f_norm, g_cal_num=1, origin_sam_bound_num=1, initial_failed_data_num=100,
                     sample_num_each_sphere=100, max_gen_times=1000, FOM_num=11, num_generate_each_norm=3, seed=0),
                  HSCS(spice=spice, f_norm=f_norm, g_var_num=1,  bound_num=1, find_MN_sam_num=100,
                        IS_sample_num=500, initial_failed_data_num=10, ratio=0.03,
                        sample_num_each_sphere=1000, FOM_num=12, seed=0),
                  MNIS(f_norm=f_norm, spice=spice, g_sam_val=1, initial_fail_num=100, initial_sample_each=500,
                        IS_num=1000, FOM_num=10, seed=0),
                  ACS(f_norm=f_norm, spice=spice, g_cal_val=1, initial_fail_num=30, initial_sample_each=1000,
                        IS_num=200, FOM_num=10, seed=0),
                 ]

    # start yield estimation
    run_models(model_list=model_list, max_num_list=10000000)

    # show results
    show_result(model_list=model_list)

