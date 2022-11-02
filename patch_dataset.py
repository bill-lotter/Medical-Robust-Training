import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image


class PatchDataset(Dataset):

    def __init__(self, path_to_images, fold, sample=0, transform=None, label_type='path'):

        self.transform = transform
        self.path_to_images = path_to_images
        #self.df = pd.read_csv("./label/cheXPert_label_.csv")
        if label_type == 'path':
            self.df = pd.read_csv("chexpert_label_Tianyu-formatted.csv")
        elif label_type == 'race':
            self.df = pd.read_csv('./labels/chexpert_race_labels-sampled.csv')
            # hacky equal sampling

        self.fold = fold
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
        elif label_type == 'race':
            self.PRED_LABEL = ['White', 'Black']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        #filename = '{0:06d}'.format(self.df.index[idx])
        filename = self.df.index[idx]
        image = Image.open(
            os.path.join(self.path_to_images, filename)  # chexpert
            #os.path.join(self.path_to_images, filename+'.png')                    # chexpert
            # os.path.join(self.path_to_images, self.fold, self.df.index[idx])      # knee  
            # os.path.join(self.path_to_images, self.df.index[idx])                 # Luna nih
            )
        image = image.convert('RGB')
        label = np.zeros(len(self.PRED_LABEL), dtype=int)
        for i in range(0, len(self.PRED_LABEL)):
             # can leave zero if zero, else make one
            if self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') > 0:
                # df.series.str.strip: remove leading and traling characters
                label[i] = self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int')
                # Becareful with the 'int' type here !!!
        if self.transform:
            image = self.transform(image)

        return (image, label)
