import pdb

import matplotlib.pyplot as plt
import tqdm
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
import pickle as pkl
from functools import partial
import torch.nn.functional as F

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


def get_results_df(all_checkpoints, check_dir, files_to_predict, im_proc_fxn, n_classes, use_softmax=False):
    results = []
    all_pred_dfs = {}
    for bn_mode in [0, 1]:
        for checkpoint in all_checkpoints:
            my_weights_path = os.path.join(check_dir, checkpoint)
            resnet = load_resnetbn(my_weights_path, n_classes)
            res_wrapper = DBNWrapper(resnet, bn_mode)

            this_pred_df = create_pred_df(res_wrapper, files_to_predict, im_proc_fxn, use_softmax)

            # idx = ~pd.isnull(this_pred_df.yhatA)
            # this_auc = roc_auc_score(this_pred_df[idx]['label'] == 'testA', this_pred_df.yhatA[idx])

            all_aucs = []
            for l in this_pred_df.label.unique():
                idx = ~pd.isnull(this_pred_df['yhat_{}'.format(l)])
                this_y = this_pred_df.loc[idx, 'label'] == l
                this_yhat = this_pred_df.loc[idx, 'yhat_{}'.format(l)]
                this_auc = roc_auc_score(this_y, this_yhat)
                all_aucs.append(this_auc)
            print(all_aucs)
            results.append([bn_mode, checkpoint, np.mean(all_aucs)])
            all_pred_dfs[(bn_mode, checkpoint)] = this_pred_df

    results_df = pd.DataFrame(results, columns=['BN_mode', 'checkpoint', 'AUC'])
    return results_df, all_pred_dfs


def create_pred_df(model, files_to_predict, im_proc_fxn, use_softmax=False):
    preds_data = []
    with torch.no_grad():
        for label in files_to_predict.keys():
            for f in tqdm.tqdm(files_to_predict[label]):
                try:
                    preds = model(im_proc_fxn(f)).squeeze()
                    if use_softmax:
                        preds = F.softmax(preds, dim=-1)
                    else:
                        preds = torch.sigmoid(preds)
                    f_preds = preds.cpu().numpy()
                    preds_data.append([f, label] + f_preds.tolist())
                except:
                    preds_data.append([f, label] + [np.nan] * len(f_preds))

    if len(f_preds) == 2:
        cols = ['yhat_Black', 'yhat_White'] #['yhatA', 'yhatB']
    else:
        cols = ['yhat_Asian', 'yhat_Black', 'yhat_White']

    pred_df = pd.DataFrame(preds_data, columns=['file', 'label'] + cols)
    return pred_df


if __name__ == '__main__':
    model_tag = 'chexpert_dsbn_res50_3race_bsz64_exclab_noclassbal_allviews'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_on_mimic = False
    use_softmax = True

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
        # data_dir = '/lotterlab/lotterb/repos/pytorch-CycleGAN-and-pix2pix/datasets/chexpert_race_v0/'
        # files_to_predict = {}
        # for label in ['testA', 'testB']:
        #     all_files = os.listdir(os.path.join(data_dir, label))
        #     files_to_predict[label] = [os.path.join(data_dir, label, f) for f in
        #                                np.random.permutation(all_files)[:n_per_label]]
        base_data_dir = '/lotterlab/datasets/'
        val_df = pd.read_csv('../../project_data/bias_interpretability/cxp_cv_splits/version_0/val.csv')
        files_to_predict = {}
        if '2race' in model_tag:
            r_tags = ['Black', 'White']
            n_classes = 2
        else:
            r_tags = ['Asian', 'Black', 'White']
            n_classes = 3
        for l in r_tags:
            fnames = val_df[val_df.Mapped_Race == l]['Path'].values
            if len(fnames) > n_per_label:
                fnames = np.random.permutation(fnames)[:n_per_label]
            files_to_predict[l] = [base_data_dir + f for f in fnames]

    base_checkpoint_dir = './checkpoint/'
    base_log_dir = './log/'

    check_dir = os.path.join(base_checkpoint_dir, model_tag)
    log_dir = os.path.join(base_log_dir, model_tag)

    all_checkpoints = os.listdir(check_dir)
    #all_checkpoints = ['checkpoint_15000.pth', 'checkpoint_20000.pth', 'checkpoint_25000.pth']
    #all_checkpoints = ['checkpoint_best.pth']
    im_proc_fxn = partial(new_preprocess_image, convert_rgb=False)
    results_df, all_pred_dfs = get_results_df(all_checkpoints, check_dir, files_to_predict, im_proc_fxn,
                                              n_classes, use_softmax)
    if test_on_mimic:
        tag = '-MIMIC'
    else:
        tag = ''

    plot_auc_by_it(results_df, os.path.join(log_dir, 'AUC_by_checkpoint_plot{}.png'.format(tag)))
    results_df.to_csv(os.path.join(log_dir, 'AUC_by_checkpoint_results_df{}.csv'.format(tag)))

    # for tup in [(0, 'checkpoint_best.pth'), (1, 'checkpoint_best.pth')]:
    #     out_path = os.path.join(log_dir, 'pred_df-bnmode{}_{}{}.csv'.format(tup[0], tup[1].replace('.pth', ''), tag))
    #     all_pred_dfs[tup].to_csv(out_path)

