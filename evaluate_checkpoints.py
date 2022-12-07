import pdb

import matplotlib.pyplot as plt
import tqdm
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
import pickle as pkl

from generate_example_adv_images import DBNWrapper, load_resnetbn, new_preprocess_image


def plot_auc_by_it(results_df, save_path):
    plt.figure()
    for bn_mode in [0, 1]:
        this_df = results_df[results_df.BN_mode == bn_mode]
        this_df = this_df[this_df.checkpoint != 'checkpoint_best.pth']
        xvals = [int(val.split('_')[1][:-4]) for val in this_df.checkpoint.values]
        yvals = this_df.AUC.values
        plt.scatter(xvals, yvals)
    plt.xlabel('Checkpoint')
    plt.ylabel('Epoch')
    plt.legend(['Std: bn=0', 'Adv: bn=1'])
    plt.savefig(save_path)
    plt.close()


def get_results_df(all_checkpoints, check_dir, files_to_predict):
    results = []
    all_pred_dfs = {}
    for bn_mode in [0, 1]:
        for checkpoint in all_checkpoints:
            my_weights_path = os.path.join(check_dir, checkpoint)
            resnet = load_resnetbn(my_weights_path, 2)
            res_wrapper = DBNWrapper(resnet, bn_mode)

            this_pred_df = create_pred_df(res_wrapper, files_to_predict)
            idx = ~pd.isnull(this_pred_df.yhatA)
            this_auc = roc_auc_score(this_pred_df[idx]['label'] == 'testA', this_pred_df.yhatA[idx])
            results.append([bn_mode, checkpoint, this_auc])
            all_pred_dfs[(bn_mode, checkpoint)] = this_pred_df

    results_df = pd.DataFrame(results, columns=['BN_mode', 'checkpoint', 'AUC'])
    return results_df, all_pred_dfs


def create_pred_df(model, files_to_predict):
    preds_data = []
    with torch.no_grad():
        for label in files_to_predict.keys():
            for f in tqdm.tqdm(files_to_predict[label]):
                try:
                    f_preds = torch.sigmoid(model(new_preprocess_image(f))).cpu().squeeze().numpy()
                    preds_data.append([f, label, f_preds[0], f_preds[1]])
                except:
                    preds_data.append([f, label, np.nan, np.nan])

    pred_df = pd.DataFrame(preds_data, columns=['file', 'label', 'yhatA', 'yhatB'])
    return pred_df


if __name__ == '__main__':
    model_tag = 'chexpert_dsbn_res50_linf-01_norm-race'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_on_mimic = False

    if test_on_mimic:
        with open('../../notebooks/bias_interpretability/mimic_exploration/mimic_formatted_file_lists.pkl', 'rb') as f:
            files_to_predict = pkl.load(f)
        files_to_predict['testA'] = files_to_predict['w']
        files_to_predict['testB'] = files_to_predict['b']
        del files_to_predict['w']
        del files_to_predict['b']
    else:
        # get some files to predict
        n_per_label = 500
        data_dir = '/lotterlab/lotterb/repos/pytorch-CycleGAN-and-pix2pix/datasets/chexpert_race_v0/'
        files_to_predict = {}
        for label in ['testA', 'testB']:
            all_files = os.listdir(os.path.join(data_dir, label))
            files_to_predict[label] = [os.path.join(data_dir, label, f) for f in
                                       np.random.permutation(all_files)[:n_per_label]]

    base_checkpoint_dir = './checkpoint/'
    base_log_dir = './log/'

    check_dir = os.path.join(base_checkpoint_dir, model_tag)
    log_dir = os.path.join(base_log_dir, model_tag)

    all_checkpoints = os.listdir(check_dir)
    #all_checkpoints = ['checkpoint_15000.pth', 'checkpoint_20000.pth', 'checkpoint_25000.pth']
    all_checkpoints = ['checkpoint_best.pth']
    results_df, all_pred_dfs = get_results_df(all_checkpoints, check_dir, files_to_predict)
    if test_on_mimic:
        tag = '-MIMIC'
    else:
        tag = ''

    #plot_auc_by_it(results_df, os.path.join(log_dir, 'AUC_by_checkpoint_plot{}.png'.format(tag)))
    #results_df.to_csv(os.path.join(log_dir, 'AUC_by_checkpoint_results_df{}.csv'.format(tag)))

    for tup in [(0, 'checkpoint_best.pth'), (1, 'checkpoint_best.pth')]:
        out_path = os.path.join(log_dir, 'pred_df-bnmode{}_{}{}.csv'.format(tup[0], tup[1].replace('.pth', ''), tag))
        all_pred_dfs[tup].to_csv(out_path)

