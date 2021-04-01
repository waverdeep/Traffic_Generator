import glob
import os
import csv


def get_all_file_path(input_dir, file_extension='csv'):
    temp = glob.glob(os.path.join(input_dir, '**', '*.{}'.format(file_extension)), recursive=True)
    return temp


def get_filename(input_filepath):
    return input_filepath.split('/')[-1]


def get_pure_filename(input_filepath):
    temp = input_filepath.split('/')[-1]
    return temp.split('.')[0]


def read_csv_file(file_path, encoding='utf-8'):
    scripts = []
    f = open(file_path, 'r', encoding=encoding)
    lines = csv.reader(f)
    for line in lines:
        scripts.append(line)
    f.close()
    return scripts


def merge_all_csv(input_dir):
    filelist = get_all_file_path(input_dir, file_extension='csv')
    merged_data = []
    need_top = True
    for file in filelist:
        data = read_csv_file(file)
        if need_top:
            merged_data += data
        else:
            del data[0]
            merged_data += data



