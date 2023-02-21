# coding=utf-8
import argparse
import json
import pprint
import os
import zipfile
from collections import defaultdict, OrderedDict
from types import SimpleNamespace
import yaml
import cv2
import numpy as np

from preprocessing.pascal_voc_utils import Reader


def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_current_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
         param_group['lr'] = lr


def load_json(fname, encoding = 'utf8'):
    with open(fname, "r", encoding= encoding) as json_file:
        d = json.load(json_file)
        return d


def save_json(fname, data, encoding = 'utf8'):
    with open(fname, "w", encoding = encoding) as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True, ensure_ascii=False)


import pickle
def save_pkl(fname, data):
    with open(fname, "wb") as f:
        pickle.dump(data, f)

def load_pkl(fname):
    with open(fname, "rb") as f:
        return pickle.load(f, encoding='utf8')



def get_label_name_map():
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'disease': 1,
        'neg':2
    }

    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    print('Label dict:')
    pprint.pprint(reverse_dict)
    return reverse_dict

def filterByconfidence(detpath, threshold, filter_class = None):
    label_dict = get_label_name_map()
    if filter_class is None:
        filter_class = list(label_dict.keys())

    for cls_index in filter_class:
        if cls_index == 0:
            continue
        detfile = os.path.join(detpath, "det_" + label_dict[cls_index] + ".txt")
        new_detfile = []
        with open(detfile, 'r') as f:
            # [img_name, score, xmin, ymin, xmax, ymax]
            lines = f.readlines()
            splitlines = [x.strip().split(' ') for x in
                          lines]

            for splitline in splitlines:
                if float(splitline[1]) >  threshold:
                    new_detfile.append(splitline)

        # rewrite to det_cls.txt file
        with open(detfile, 'w') as f:
            for a_det in new_detfile:
                f.write(' '.join(a_det) + '\n')



import sys
sys.path.append('/')
sys.path.append('../../../')
import os.path as osp
def bbox_counter(path):
    counter = defaultdict(int)
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('xml'):

                objs = Reader(osp.join(root, filename)).get_objects()
                for label in objs['category_name']:
                    counter[label] += 1
    print(counter)


def get_format_time(timezone_str = 'Asia/Seoul', time_format = '%Y%m%d-%H%M%S'):
    from pytz import timezone
    from datetime import datetime
    KST = timezone(timezone_str)
    return datetime.now(tz = KST).strftime(time_format)

def save_hyperparams(path, context):
    with open(osp.join(path, 'hyper_params.txt'), 'a+') as file:
        file.write("%s\n"%get_format_time(time_format='%Y-%m-%d %H:%M:%S'))
        file.write(context)
        file.write('%s%s'%('='*80, '\n'))
    return osp.join(path, 'hyper_params.txt')


def yaml2hyperparams(args):
    if osp.exists(args.config):
        opt = vars(args)
        args = yaml.load(open(args.config), Loader = yaml.FullLoader)
        opt.update(args)
        args = argparse.Namespace(**opt)

    return args

def backup(path, base_path = None, suffixs = ['py', 'yaml'], exclude_folder_names = ['output', '.idea']):
    base_path = osp.join(osp.dirname(osp.abspath(__file__)), '../../../') if base_path is None else base_path

    zipf = zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED)
    for root, _, filenames in os.walk(base_path):
        r = osp.abspath(root)
        if sum([e in r for e in exclude_folder_names]) == 0:
            for filename in filenames:
                if filename.split('.')[-1] in suffixs:
                    zipf.write(os.path.join(root, filename),
                               os.path.relpath(os.path.join(root, filename),
                                               os.path.join(base_path, '../../..')))
    zipf.close()


def namespace2dict(input):
    if isinstance(input, SimpleNamespace):
        input = vars(input)
        for key, value in input.items():
            input[key] = namespace2dict(value)
        return input
    elif isinstance(input, list):
        for index, v in enumerate(input):
            input[index] = namespace2dict(v)
        return input
    else:
        return input

def draw_box(img, bb, name, color = (0, 0, 255)):

    bb = [int(float(b)) for b in bb]
    img = cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), color, 1)

    text_pos = [bb[0], bb[1] - 10]
    # 'xmin', 'ymin', 'xmax', 'ymax'
    if text_pos[1] < 0:
        text_pos = [bb[2], bb[3] - 10]
    img = cv2.putText(img, str(name), tuple(text_pos),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return img

def draw_bboxes(img, obj_dicts, color = (0, 0, 255)):
    assert np.max(img) > 1.0
    img = np.array(img)
    for box, name in zip(obj_dicts['bboxes'], obj_dicts['name']):
        img = draw_box(img, box, name, color = color)
    return img

if __name__ == '__main__':
    pass