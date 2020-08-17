#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Author: Duanzhixiang(zhixiangduan@deepmotion.ai)
from __future__ import division

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')

    parser.add_argument('--out_path',
                        help='needed by job client')
    parser.add_argument('--in_path',
                        help='needed by job client')
    parser.add_argument('--pretrained_path', help='needed by job client')
    parser.add_argument('--job_name', help='needed by job client')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    print(os.system('kill -9 40450 40451 40452 40453 40454 40455 40448'))
    # print(os.system('cp /node01_data5/StereoMatching/SceneFlow /data2/ -r'))
    # print(os.system('tree -L 1 /data2/SceneFlow'))
    print('in path is ', args.in_path)
    print(os.system('top -b | head -n 15'))
    print(os.system('ls /data1'))
    print(os.system('df -h'))
    print(os.system('nvidia-smi'))
    print(os.system('nvcc --version'))


if __name__ == '__main__':
    main()


