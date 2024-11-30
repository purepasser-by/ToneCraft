# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#


import os
import io
import shutil
import argparse

import numpy as np
import json

import dtw







def cal_md(targets, hypos):
    # flatten_hypos = list(map(flatten, hypos))
    # flatten_targets = list(map(flatten, targets))
    #
    # flatten_hypo_samples = list(map(sample_notes, flatten_hypos))
    # flatten_target_samples = list(map(sample_notes, flatten_targets))

    dtw_mean = []
    for i in range(len(targets)):
        if len(targets[i]) == 0 or len(hypos[i]) == 0:
            continue

        d1 = np.array(targets[i]).reshape(-1, 1)
        d2 = np.array(hypos[i]).reshape(-1, 1)

        d1 = d1 - np.mean(d1)
        d2 = d2 - np.mean(d2)
        d, _, _, _ = dtw.accelerated_dtw(d1, d2, dist='euclidean')
        dtw_mean.append(d / len(d2))

    return sum(dtw_mean) / len(dtw_mean)



def load_hypos(file_path):
    #读json文件
    with open(file_path, 'r') as file:
        data = json.load(file)

    pitch_list = []
    for record in data:
        if 'output' in record:
            pitch_list.append(record['output'])

    return pitch_list

def load_targets(file_path):
    #读json文件
    with open(file_path, 'r') as file:
        data = json.load(file)

    pitch_list = []
    for record in data:
        if 'pitch' in record:
            pitch_list.append(record['pitch'])

    return pitch_list


def load_gen_tsk(hypos_path,targets_path):
    # 读json文件
    with open(hypos_path, 'r') as file:
        data = json.load(file)

    hypos_list = []
    for record in data:
        if 'output' in record:
            hypos_list.append(record['output'])

    # 读json文件
    with open(targets_path, 'r') as file:
        data = json.load(file)

    targets_list = []
    for record in data:
        if 'pitch' in record:
            targets_list.append(record['pitch'])

    return hypos_list,targets_list


def load_msk_tsk(hypos_path,targets_path):
    # 读json文件
    with open(hypos_path, 'r') as file:
        data = json.load(file)

    hypos_list = []
    for record in data:
        if 'pitch' in record:
            pitch_list = record['pitch']
            # mask替换
            for i, idx in enumerate(record['mask_index']):
                pitch_list[idx] = record['output'][i]#i 测试评价指标
            hypos_list.append(pitch_list)

    # 读json文件
    with open(targets_path, 'r') as file:
        data = json.load(file)

    targets_list = []
    for record in data:
        if 'pitch' in record:
            targets_list.append(record['pitch'])

    return hypos_list,targets_list

