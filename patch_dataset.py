import pdb

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image


class PatchDataset(Dataset):

    def __init__(self, path_to_images, fold, sample=0, transform=None, label_type='path',
                 samples_per_epoch=10000, equal_sampling=True, exclusive_labels=False, filter_views=True):
 
        self.transform = transform
        self.path_to_images = path_to_images
        self.label_type = label_type
        self.samples_per_epoch = samples_per_epoch
        self.equal_sampling = equal_sampling
        self.exclusive_labels = exclusive_labels
        #self.df = pd.read_csv("./label/cheXPert_label_.csv")
        if label_type == 'path':
            self.df = pd.read_csv("chexpert_label_Tianyu-formatted.csv")
        elif 'race' in label_type:
            #self.df = pd.read_csv('./labels/chexpert_race_labels-sampled.csv') # hacky equal sampling
            self.df = pd.read_csv('../../project_data/bias_interpretability/cxp_cv_splits/version_0/{}.csv'.format(fold.replace('valid', 'val')))
            print('filter_views', filter_views)
            if filter_views:
                keep_idx = (self.df['Frontal/Lateral'] == 'Frontal') & self.df['AP/PA'].isin(['AP', 'PA'])
                self.df = self.df[keep_idx].copy()

        self.fold = fold
        if label_type == 'path':
            # the 'fold' column says something regarding the train/valid/test seperation
            self.df = self.df[self.df['fold'] == fold]
            if(sample > 0 and sample < len(self.df)):
                self.df = self.df.sample(frac=sample, random_state=42)
                print('subsample the training set with ratio %f' % sample)
        
        # self.df = self.df.set_index('scan index')
        #self.df = self.df.set_index('Image Index')
        self.df = self.df.set_index('Path')
        # df.set_index: set the dataframe index using existing columns. 
        # self.PRED_LABEL = ['malignancy']
        # self.PRED_LABEL = ['healthy', 'partially injured', 'completely ruptured']
        if label_type == 'path':
            self.PRED_LABEL = ['No Finding', 'Cardiomegaly', 'Edema',
                                'Consolidation', 'Pneumonia', 'Atelectasis',
                                'Pneumothorax', 'Pleural Effusion']
            self.n_labels = len(self.PRED_LABEL)
        elif 'race' in label_type:
            self.n_labels = int(label_type[-1])
            if self.n_labels in [1, 2]:
                self.PRED_LABEL = ['Black', 'White']
            elif self.n_labels == 3:
                self.PRED_LABEL = ['Asian', 'Black', 'White'] #['White', 'Black']
            self.df = self.df[self.df.Mapped_Race.isin(self.PRED_LABEL)].copy()

            label_idx_map = {}
            for i, r in enumerate(self.PRED_LABEL):
                label_idx_map[r] = i
            self.df['label_idx'] = self.df.Mapped_Race.map(label_idx_map)
            self.idxs_per_label = {}
            for i in range(len(self.PRED_LABEL)):
                self.idxs_per_label[i] = np.where(self.df.label_idx == i)[0]

    def __len__(self):
        # if self.label_type == 'race':
        #     return self.samples_per_epoch
        # else:
        return len(self.df)

    def __getitem__(self, idx):
        #filename = '{0:06d}'.format(self.df.index[idx])
        if self.equal_sampling:
            idx = np.random.randint(len(self.PRED_LABEL))
            idx = np.random.choice(self.idxs_per_label[idx])

        filename = self.df.index[idx]
        image = Image.open(
            os.path.join(self.path_to_images, filename)  # chexpert
            #os.path.join(self.path_to_images, filename+'.png')                    # chexpert
            # os.path.join(self.path_to_images, self.fold, self.df.index[idx])      # knee  
            # os.path.join(self.path_to_images, self.df.index[idx])                 # Luna nih
            )
        #image = image.convert('RGB')

        if self.exclusive_labels:
            label = np.zeros(1, dtype=int)
        else:
            label = np.zeros(len(self.PRED_LABEL), dtype=int)

        if self.label_type == 'path':
            for i in range(0, len(self.PRED_LABEL)):
                 # can leave zero if zero, else make one
                if self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') > 0:
                    # df.series.str.strip: remove leading and traling characters
                    label[i] = self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int')
                    # Becareful with the 'int' type here !!!
        else:
            # race = self.df.Mapped_Race.iloc[idx]
            # r_idx = [i for i in range(len(self.PRED_LABEL)) if (race == self.PRED_LABEL[i])]
            # assert len(r_idx) == 1
            # r_idx = r_idx[0]
            # label[r_idx] = 1
            l_idx = self.df.label_idx.iloc[idx]
            if self.n_labels == 1 or self.exclusive_labels:
                label[0] = l_idx
            else:
                label[l_idx] = 1

        if self.transform:
            image = self.transform(image)

        return (image, label)
