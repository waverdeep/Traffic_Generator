from torch.utils.data import Dataset, DataLoader
import glob
import os
import natsort
import pandas as pd
import torch

'''
Dataset
* AmazonPrime
 - Driving : 움직이면서 하는 것
 - Static : 가만히 서있으면서 하는 것 
'''


def get_all_file_path(input_dir, file_extension='csv'):
    temp = glob.glob(os.path.join(input_dir, '**', '*.{}'.format(file_extension)), recursive=True)
    temp = natsort.natsorted(temp)
    return temp


class AmazonPrimeDataset(Dataset):
    def __init__(self, input_dir):
        self.filelist = get_all_file_path(input_dir, file_extension='csv')

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        item = pd.read_csv(self.filelist[idx])
        item = torch.tensor(item['DL_bitrate'].values, dtype=torch.float32)
        # item = item.unsqueeze(0)
        return item



