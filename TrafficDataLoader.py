import glob
import os

'''
Dataset
* AmazonPrime
 - Driving : 움직이면서 하는 것
 - Static : 가만히 서있으면서 하는 것 
'''


def get_all_file_path(input_dir, file_extension='csv'):
    temp = glob.glob(os.path.join(input_dir, '**', '*.{}'.format(file_extension)), recursive=True)
    return temp


def get_filename(input_filepath):
    return input_filepath.split('/')[-1]


def get_pure_filename(input_filepath):
    temp = input_filepath.split('/')[-1]
    return temp.split('.')[0]

