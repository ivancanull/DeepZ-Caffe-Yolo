#!/usr/bin/env python2.7
# coding=utf-8

import os


if __name__ == '__main__':
    data_root = '/home/djn/projects/DAC-Contest/DACSDC-DeepZ/Train/data/dataset'
    train_set_path = os.path.join(data_root, 'train_dataset.txt')
    valid_set_path = os.path.join(data_root, 'valid_dataset.txt')

    # retrieve data
    with open(train_set_path, 'r') as f:
        train_set = [line.strip() for line in f]
    with open(valid_set_path, 'r') as f:
        valid_set = [line.strip() for line in f]

    train_lst = []
    for frame_path in train_set:
        cls = frame_path.split(os.path.sep)[-2]
        frame = frame_path.split(os.path.sep)[-1].split('.')[0]
        image_path = os.path.join(cls, frame + '.jpg')
        label_path = os.path.join(cls, frame + '.txt')
        train_lst.append('%s %s' % (image_path, label_path))
    valid_lst = []
    for frame_path in valid_set:
        cls = frame_path.split(os.path.sep)[-2]
        frame = frame_path.split(os.path.sep)[-1].split('.')[0]
        image_path = os.path.join(cls, frame + '.jpg')
        label_path = os.path.join(cls, frame + '.txt')
        valid_lst.append('%s %s' % (image_path, label_path))

    # write to file
    with open('meta_data_train.txt', 'w') as f:
        f.write('\n'.join(train_lst))
    with open('meta_data_valid.txt', 'w') as f:
        f.write('\n'.join(valid_lst))
