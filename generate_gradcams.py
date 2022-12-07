import sys
import os
import torchvision
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
sys.path.insert(0, '../SVCEs_code/')

from generate_example_adv_images import load_resnetbn, DBNWrapper, new_preprocess_image
from visualization.gradcam import ModelOutputs, GradCam


def initialize_resnet_from_resnetdbn(res_model, resdbn_model, bn_mode=1):
    with torch.no_grad():
        res_params = res_model.state_dict()
        dbn_params = resdbn_model.state_dict()
        for key in res_params:
            if 'bn' in key or ('downsample' in key and key not in dbn_params):
                this_str = 'bn' if 'bn' in key else 'downsample'
                n_char = len(this_str)
                idx = key.find(this_str)
                if this_str == 'bn':
                    new_key = key[:idx + n_char + 2] + 'bns.' + str(bn_mode) + key[idx + n_char + 1:]
                else:
                    new_key = key[:idx + n_char + 3] + 'bns.' + str(bn_mode) + key[idx + n_char + 2:]
                new_param = dbn_params[new_key]
            else:
                new_param = dbn_params[key]
            res_params[key] = new_param

        res_model.load_state_dict(res_params)


def load_models(checkpoint_path):
    resnet_dbn = load_resnetbn(checkpoint_path, n_classes=2)

    resnets = {}
    for bn_mode in [0, 1]:
        resnet = torchvision.models.resnet50(num_classes=2)
        resnet.eval()
        resnet = resnet.to('cuda')
        initialize_resnet_from_resnetdbn(resnet, resnet_dbn, bn_mode)
        resnets[bn_mode] = resnet

    return resnets


def plot_cam_on_image(img, mask, out_path, size=(4, 4), dpi=80):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_MAGMA)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.moveaxis(np.float32(img.cpu().squeeze()), 0, -1)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)

    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap('gray')
    ax.imshow(cam, aspect='equal')
    plt.savefig(out_path, dpi=dpi)
    plt.close()


if __name__ == '__main__':
    model_tag = 'chexpert_dsbn_res50_linf-01_norm-race_b64'
    checkpoint_tag = 'checkpoint_25000'
    generate_average = True

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    n_files_per_class = 100  # 50

    base_checkpoint_dir = '/lotterlab/lotterb/repos/Medical-Robust-Training/checkpoint/'
    checkpoint_path = os.path.join(base_checkpoint_dir, model_tag, checkpoint_tag + '.pth')

    resnets = load_models(checkpoint_path)

    data_dir = '/lotterlab/lotterb/repos/pytorch-CycleGAN-and-pix2pix/datasets/chexpert_race_v0/'
    files_to_predict = {}
    for label in ['testA', 'testB']:
        all_files = os.listdir(os.path.join(data_dir, label))
        files_to_predict[label] = [os.path.join(data_dir, label, f) for f in
                                   np.random.permutation(all_files)[:n_files_per_class]]

    base_out_dir = '/lotterlab/lotterb/project_data/bias_interpretability/Medical-Robust-Training/gradcam_images_plots/'
    if not os.path.exists(base_out_dir):
        os.mkdir(base_out_dir)

    out_dir = os.path.join(base_out_dir, model_tag + '-' + checkpoint_tag)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    cam_models = {bn_mode: GradCam(resnets[bn_mode], feature_module=resnets[bn_mode].layer4,
                                   target_layer_names=["2"], use_cuda=True) for bn_mode in resnets}

    if generate_average:
        averages = {}
        counts = {}
        for label in files_to_predict:
            averages[label] = {bn_mode: {class_num: None for class_num in [0, 1]} for bn_mode in [0, 1]}
            counts[label] = {bn_mode: {class_num: 0 for class_num in [0, 1]} for bn_mode in [0, 1]}
            for f in files_to_predict[label]:
                x = new_preprocess_image(f)
                for bn_mode in [0, 1]:
                    for target_class in [0, 1]:
                        grayscale_cam, pred_lab = cam_models[bn_mode](x, target_category=target_class, categories=[0, 1])
                        if np.std(grayscale_cam): # filter out nans
                            c = counts[label][bn_mode][target_class]
                            curr_a = averages[label][bn_mode][target_class]
                            if c:
                                averages[label][bn_mode][target_class] = (1. / (c + 1)) * grayscale_cam + (c / (c + 1)) * curr_a
                            else:
                                averages[label][bn_mode][target_class] = grayscale_cam
                            counts[label][bn_mode][target_class] = c + 1

        for label in files_to_predict:
            for bn_mode in [0, 1]:
                for target_class in [0, 1]:
                    out_name = 'averagecam_{}_bn{}_targetclass{}.png'.format(label, bn_mode, target_class)

                    av_im = averages[label][bn_mode][target_class]
                    av_im = (av_im - av_im.min()) / (av_im.max() - av_im.min())

                    fig = plt.figure()
                    fig.set_size_inches((4, 4))
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    plt.set_cmap('gray')
                    ax.imshow(av_im, aspect='equal')
                    plt.savefig(os.path.join(out_dir, out_name), dpi=80)
                    plt.close()
    else:
        count = 0
        out_data = []
        for label in files_to_predict:
            for f in files_to_predict[label]:
                x = new_preprocess_image(f)
                f_scores = {}
                for bn_mode in [0, 1]:
                    grayscale_cam, pred_lab = cam_models[bn_mode](x, categories=[0, 1])
                    out_name = '{}_{}_bn{}_predclass{}.png'.format(label, count, bn_mode, pred_lab)
                    plot_cam_on_image(x, grayscale_cam, os.path.join(out_dir, out_name))
                    with torch.no_grad():
                        f_scores[bn_mode] = torch.sigmoid(resnets[bn_mode](x)).cpu().squeeze().numpy()

                count += 1
                out_data.append([f, label, f_scores[0][0], f_scores[0][1], f_scores[1][0], f_scores[1][1]])

        columns = ['file_path', 'label', 'bnmode0_A_score', 'bnmode0_B_score', 'bnmode1_A_score', 'bnmode1_B_score']
        out_df = pd.DataFrame(out_data, columns=columns)

        out_df.to_csv(os.path.join(out_dir, 'prediction_scores.csv'))