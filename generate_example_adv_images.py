import pdb
import sys
import torchvision as tv
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage.transform import resize
import tqdm
import pandas as pd
import os
import torch.nn.functional as F

sys.path.insert(0, '../SVCEs_code/')

from utils.train_types.helpers import create_attack_config, get_adversarial_attack

from model.resnetdsbn import resnet50dsbn


class DBNWrapper(torch.nn.Module):
    def __init__(self, orig_model, bn_index=0, add_softmax=False):
        super().__init__()
        self.orig_model = orig_model
        self.bn_index = bn_index
        self.add_softmax = add_softmax

    def forward(self, x):
        x = self.orig_model.forward(x, [self.bn_index])
        if self.add_softmax:
            x = F.softmax(x, dim=-1)
        return x


def load_resnetbn(check_path, n_classes=8):
    model = resnet50dsbn()
    model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
    checkpoint = torch.load(check_path)
    model.load_state_dict(checkpoint)
    model.eval()
    model = model.to('cuda')
    return model


def new_preprocess_image(image_path, convert_rgb=True):
    img = Image.open(image_path)
    if convert_rgb:
        img = img.convert('RGB')
    transform_test = tv.transforms.Compose([
        tv.transforms.Resize(224),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
    ])

    return transform_test(img).unsqueeze(0).to('cuda')


def create_adv_im(model_to_attack, attack_class, x, eps=75, steps=75, stepsize=0.1,
                  norm='l1.5', attack_type='afw', loss='log_conf', num_classes=8):
    attack_config = create_attack_config(eps, steps, stepsize, norm,
                                         pgd=attack_type, normalize_gradient=True)

    att = get_adversarial_attack(attack_config, model_to_attack, loss, num_classes)

    y = np.zeros((1), int)
    y[0] = attack_class
    y = torch.from_numpy(y).to('cuda')
    with torch.no_grad():
        if norm == 'l1.5':
            x_adv = att.perturb(x, y, targeted=True).detach()
        else:
            x_adv = att.perturb(x, y, best_loss=True)[0]
            x_adv = x_adv.detach()

    return x_adv


def reformat_x(x_in):
    x_in = torch.clamp(x_in, 0, 1)
    img_new = x_in.cpu().numpy()
    #img_new = (img_new - img_new.min())/(img_new.max() - img_new.min())
    if img_new.ndim == 4:
        return img_new[0,0]
    else:
        return img_new[0]


def plot_and_save_image(im, out_path, size=(4, 4), dpi=80):
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap('gray')
    ax.imshow(im, aspect='equal')
    plt.savefig(out_path, dpi=dpi)
    plt.close()


if __name__ == '__main__':
    model_tag = 'chexpert_dsbn_res50_2race_bsz64_exclab_noclassbal_allviews'
    checkpoint_tag = 'checkpoint_5000'
    #model_tag = 'chexpert_dsbn_res50_linf-01_norm-race'
    #checkpoint_tag = 'checkpoint_best'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    n_files_per_class = 5 #50
    add_softmax = False
    eps = 25

    base_checkpoint_dir = '/lotterlab/lotterb/repos/Medical-Robust-Training/checkpoint/'
    checkpoint_path = os.path.join(base_checkpoint_dir, model_tag, checkpoint_tag + '.pth')

    n_classes = 3 if '3race' in model_tag else 2
    resnet = load_resnetbn(checkpoint_path, n_classes=n_classes)
    res_wrapper = DBNWrapper(resnet, bn_index=1, add_softmax=add_softmax)  # use adv bn_index

    base_data_dir = '/lotterlab/datasets/'
    val_df = pd.read_csv('../../project_data/bias_interpretability/cxp_cv_splits/version_0/val.csv')
    files_to_predict = {}
    if '2race' in model_tag:
        r_tags = ['Black', 'White']
    else:
        r_tags = ['Asian', 'Black', 'White']
    for l in r_tags:
        fnames = val_df[val_df.Mapped_Race == l]['Path'].values
        fnames = np.random.permutation(fnames)[:n_files_per_class]
        files_to_predict[l] = [base_data_dir + f for f in fnames]

    # data_dir = '/lotterlab/lotterb/repos/pytorch-CycleGAN-and-pix2pix/datasets/chexpert_race_v0/'
    # files_to_predict = {}
    # for label in ['testA', 'testB']:
    #     all_files = os.listdir(os.path.join(data_dir, label))
    #     files_to_predict[label] = [os.path.join(data_dir, label, f) for f in
    #                                np.random.permutation(all_files)[:n_files_per_class]]

    base_out_dir = '/lotterlab/lotterb/project_data/bias_interpretability/Medical-Robust-Training/SVCE_adv_images_plots/'
    out_dir = os.path.join(base_out_dir, model_tag + '-' + checkpoint_tag)
    if add_softmax:
        out_dir += 'with-softmax'
    if eps != 75:
        out_dir += 'eps-{}'.format(eps)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    plot_dir = os.path.join(out_dir, 'plots')
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    out_data = []
    count = 0
    for label in files_to_predict:
        label_out_dir = os.path.join(plot_dir, label)
        if not os.path.exists(label_out_dir):
            os.mkdir(label_out_dir)
        for f in tqdm.tqdm(files_to_predict[label]):
            x = new_preprocess_image(f, convert_rgb=False)

            adv_ims = {class_num: create_adv_im(res_wrapper, class_num, x, num_classes=n_classes, eps=eps) for class_num in range(n_classes)}
            with torch.no_grad():
                if add_softmax:
                    orig_preds = res_wrapper(x).cpu().squeeze().numpy()
                    adv_preds = {class_num: res_wrapper(adv_ims[class_num]).cpu().squeeze().numpy() for
                                 class_num in range(n_classes)}
                else:
                    orig_preds = torch.sigmoid(res_wrapper(x)).cpu().squeeze().numpy()
                    adv_preds = {class_num: torch.sigmoid(res_wrapper(adv_ims[class_num])).cpu().squeeze().numpy() for class_num in range(n_classes)}

            # out_data.append([f, label, orig_preds[0], orig_preds[1], adv_preds[0][0], adv_preds[0][1],
            #                  adv_preds[1][0], adv_preds[1][1]])

            out_data.append([f, label] + orig_preds.tolist() + adv_preds[0].tolist() + adv_preds[1].tolist()) # + adv_preds[2].tolist())

            orig_im = reformat_x(x)
            for class_num, im in adv_ims.items():
                im = reformat_x(im)
                im_save_path = os.path.join(label_out_dir, '{}_adv{}.jpg'.format(count, class_num))
                plot_and_save_image(im, im_save_path)

                im_save_path_diff = os.path.join(label_out_dir, '{}_adv{}-diff.jpg'.format(count, class_num))
                plot_and_save_image(im - orig_im, im_save_path_diff)

            im_save_path_orig = os.path.join(label_out_dir, '{}_orig.jpg'.format(count))
            plot_and_save_image(orig_im, im_save_path_orig)

            count += 1

    #columns = ['file_path', 'label', 'orig_A_score', 'orig_B_score', 'advA_A_score', 'advA_B_score', 'advB_A_score', 'advB_B_score']
    columns = ['file_path', 'label']
    for t in ['orig', 'adv0', 'adv1']: #, 'adv2']:
        for v in range(n_classes):
            columns += [t + '_' + str(v) + '_score']
    out_df = pd.DataFrame(out_data, columns=columns)

    out_df.to_csv(os.path.join(out_dir, 'adv_prediction_scores.csv'))





