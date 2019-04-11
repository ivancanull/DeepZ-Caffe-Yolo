#!/usr/bin/env python2.7
# coding=utf-8

import os


if __name__ == '__main__':
    data_root = '~/Project/DeepZ-Caffe-Yolo/'
    train_set_path = os.path.join(data_root, 'meta_data_train_1.txt')
    valid_set_path = os.path.join(data_root, 'meta_data_valid_1.txt')

    # retrieve data
    with open(train_set_path, 'r') as f:
        train_set = [line.strip() for line in f]
    with open(valid_set_path, 'r') as f:
        valid_set = [line.strip() for line in f]

    train_lst = []
    for item in train_set:
        if item.find('bird1') != 0 and item.find('car7') != 0 and item.find('group1') != 0:
            train_lst.append(item)

    valid_lst = []
    for item in valid_set:
        if item.find('bird1') != 0 and item.find('car7') != 0 and item.find('group1') != 0:
            valid_lst.append(item)

    # write to file
    with open('meta_data_train.txt', 'w') as f:
        f.write('\n'.join(train_lst))
    with open('meta_data_valid.txt', 'w') as f:
        f.write('\n'.join(valid_lst))