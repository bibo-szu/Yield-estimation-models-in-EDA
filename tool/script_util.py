import pandas as pd
import numpy as np
import os
from tool.util import get_model_class_name, print_metrics_v2

def delete_res_dir(model_list):
    for model in model_list:
        model_name = get_model_class_name(model)
        dir_path = f"./results/{model_name}_case{model.spice.case}_{model.seed}.csv"
        if os.path.exists(dir_path) == True:
            os.remove(dir_path)

def run_models(model_list, max_num_list):
    delete_res_dir(model_list)

    if isinstance(type(max_num_list),type(int)):
        max_num_list = [max_num_list] * len(model_list)

    for i in range(len(model_list)):
        model = model_list[i]
        model.start_estimate(max_num=max_num_list[i])

def show_result(model_list):

    result = dict()
    MC_idx = -1
    for i,model in enumerate(model_list):
        model_name = get_model_class_name(model)
        dir_path = f"./results/{model_name}_case{model.spice.case}_{model.seed}.csv"
        if(model_name=="MC"):
            MC_idx=i
            metrics = dict()
            metrics['Pfail'] = np.array(pd.read_csv(dir_path)['Pfail'])[-1]
            metrics['Num'] = np.array(pd.read_csv(dir_path)['num'])[-1]
            metrics['Speedup'] = '{:.3f} X'.format(1.)
            metrics['Error'] = '{:.3f} %'.format(0.)
            metrics['Success'] = 'Y'
            result[model_name]=metrics

    for i,model in enumerate(model_list):
        model_name = get_model_class_name(model)
        dir_path = f"./results/{model_name}_case{model.spice.case}_{model.seed}.csv"

        if(model_name!="MC"):
            metrics = dict()
            metrics['Pfail'] = np.array(pd.read_csv(dir_path)['Pfail'])[-1]
            metrics['Num'] = np.array(pd.read_csv(dir_path)['num'])[-1]
            if MC_idx!=-1:
                error = np.abs(metrics['Pfail']-result['MC']['Pfail'])/result['MC']['Pfail']
                metrics['Speedup'] = '{:.3f} X'.format(result['MC']['Num']/metrics['Num'])
                metrics['Error'] = '{:.3f} %'.format(error*100)
                if error < 0.8:
                    metrics['Success'] = 'Y'
                else:
                    metrics['Success'] = 'N'
            else:
                metrics['Speedup'] = 'N/A'
                metrics['Error'] = 'N/A'
                metrics['Success'] = 'N/A'
            result[model_name]=metrics

    print_metrics_v2(metrics=result, metric_names=['Pfail','Num','Speedup','Error','Success'])

def import_data_new(case, model_name, seed, metric):
    df = np.array(pd.read_csv(f"./results/{model_name}_case{case}_{seed}.csv"))
    if metric == "Pfail":
        return df[:,0]
    elif metric == "FOM":
        return df[:,1]
    elif metric == "num":
        return df[:,2]