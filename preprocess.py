import glob
import os
import natsort
import pandas as pd


def get_all_file_path(input_dir, file_extension='csv'):
    temp = glob.glob(os.path.join(input_dir, '**', '*.{}'.format(file_extension)), recursive=True)
    temp = natsort.natsorted(temp)
    return temp


def get_filename(input_filepath):
    return input_filepath.split('/')[-1]


def get_pure_filename(input_filepath):
    temp = input_filepath.split('/')[-1]
    temp = temp.replace('.csv', '')
    return temp


def get_train_test_csv_data(input_dir, output_dir, sequence_length=1260):
    filelist = get_all_file_path(input_dir, file_extension='csv')
    for idx, file in enumerate(filelist):
        pure_name = get_pure_filename(file)
        lines = pd.read_csv(file)
        for i in range(len(lines)-(sequence_length-1)):
            partition = lines[i:i+(sequence_length-1)]
            partition.to_csv('{}{}-re{}.csv'.format(output_dir, pure_name, i), mode='w', index=False)


if __name__ == '__main__':
    get_train_test_csv_data('dataset/Amazon_Prime/Static/', 'dataset/reformat_amazon/static/')



