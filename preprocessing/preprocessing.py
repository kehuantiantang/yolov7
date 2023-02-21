# coding=utf-8
import os.path as osp
import os
import random

import cv2
from tqdm import tqdm

import pandas as pd

from preprocessing.json_polygon import JsonLoader
from preprocessing.pascal_voc_utils import Writer
from utils.logger import Logger


def read_dir(path, rename=True):
    '''
    read the file from directory
    :param path:
    :return:
    '''
    if rename:
        for root, _, filenames in os.walk(path):
            for filename in tqdm(sorted(filenames), desc='Rename:'):
                if ' ' in filename:
                    os.rename(osp.join(root, filename), osp.join(root, filename.replace(' ', '_')))
        print('Rename all the files and remove exception characters')

    files = {}
    for root, _, filenames in os.walk(path):
        for filename in tqdm(sorted(filenames), desc='Find image json file:'):
            if filename.endswith('json'):
                name = filename.split('.')[0]
                files[name] = osp.join(root, name)

    return files

def obj_to_xml(xml_path, objects):
    writer = Writer(objects['filename'], objects['width'], objects['height'], database = objects['filename'])
    writer.addBboxes(objects['bboxes'], objects['category_name'])
    writer.save(xml_path)

def mk_dirs(target_path, folder):
    # json include polygon, point
    # image gis image
    img_target = osp.join(target_path, folder, 'img')
    # segmentation mask
    mask_target = osp.join(target_path, folder, 'mask')
    # check whether the polygon and bbox is correctly annotate
    vis_target = osp.join(target_path, folder, 'vis')
    # bbox xml
    xml_target = osp.join(target_path, folder, 'xml')

    if not osp.exists(img_target):
        os.makedirs(img_target, exist_ok=True)
    if not osp.exists(vis_target):
        os.makedirs(vis_target, exist_ok=True)
    if not osp.exists(mask_target):
        os.makedirs(mask_target, exist_ok=True)
    if not osp.exists(xml_target):
        os.makedirs(xml_target, exist_ok=True)

    return img_target, mask_target, vis_target, xml_target


def preprocess_train_data(paths, args, preprocess_folder_name = 'preprocess', suffix = '.tif', split_rate = -1,
                          seed = 0, dsize = 800):
    """

    :param path:
    :param args:
    :param preprocess_folder_name:
    :param suffix:
    :param split_rate:
    :param seed:
    :param dsize:
    :return:
    """
    assert isinstance(paths, list) and len(paths) == 2, 'Train should have to input folder' %paths
    assert sum([osp.exists(p) for p in paths]) == 2, 'Current dataset path %s is not exists!' %paths

    Logger.info('Start to preprocess the dataset in path %s, please wait a moment'%paths[0], '.'*20)
    root_path, data_name = osp.split(paths[0])
    # should be not hard path

    preprocess_path = osp.join(args.voc, '%s_%s'%(preprocess_folder_name, data_name))
    # preprocess_path = osp.join('/Users/sober/Downloads/test_dataset')
    os.makedirs(preprocess_path, exist_ok =True)


    args.num_fold = 0
    args.fold_path = osp.join(preprocess_path, args.mode, 'fold.csv')
    target_path = osp.join(preprocess_path, args.mode)

    args.train_xml = osp.join(preprocess_path, args.mode, 'train', 'xml')

    args.input = target_path

    fold_records = {'fold':[], 'image_id':[]}
    jl = JsonLoader()

    if not osp.exists(args.input):
        for status, path in zip(['train', 'val'], paths):
            img_target, mask_target, vis_target, xml_target = mk_dirs(target_path, status)
            if status == 'train':
                fold_id = 0
            else:
                fold_id = 1


            file_path = read_dir(path)
            disease_counter, dis_img_counter, no_dis_img_counter, total_img = 0, 0, 0, 0
            for name, path in tqdm(file_path.items(), desc='Convert to voc format:'):

                jpg_path = '%s.%s'%(path.strip('.'), suffix.strip('.'))
                gt_path = path + '_gt.jpg'
                json_path = path + '.json'


                assert  osp.exists(jpg_path), '%s is not exist' %jpg_path
                jpg_img = cv2.imread(jpg_path)
                # if dsize > 0:
                #     jpg_img = cv2.resize(jpg_img, (dsize, dsize))


                context = jl.load_json(json_path)
                attributes = jl.get_objects(context)
                # if dsize > 0:
                #     attributes = jl.resize_annotation(attributes, dsize)

                nb_disease = len(attributes['polygons'])

                disease_counter += nb_disease
                if nb_disease > 0:
                    dis_img_counter += 1
                else:
                    no_dis_img_counter += 1
                total_img += 1


                if nb_disease > 0:
                    # draw bbox image
                    # jpg_img_boxes = jl.draw_bboxes(jpg_img, attributes)
                    # draw polygon image
                    # jpg_img_polygons = jl.draw_polygons(jpg_img, attributes)
                    # draw image
                    # jpg_mask = jl.draw_mask(jpg_img, attributes, color_dict= {'disease':(1, 1, 1)}, single_channel= True)


                    cv2.imwrite(osp.join(img_target, name.replace(' ', '_') + '_%02d.jpg'%nb_disease), jpg_img)
                    # cv2.imwrite(osp.join(vis_target, name.replace(' ', '_') + '_%02d.jpg'%nb_disease), cv2.hconcat([jpg_img_boxes, jpg_img_polygons]))
                    # cv2.imwrite(osp.join(mask_target, name.replace(' ', '_') + '_%02d.png'%nb_disease), jpg_mask)

                    obj_to_xml(osp.join(xml_target, name.replace(' ', '_') + '_%02d.xml'%nb_disease), attributes)

                    fold_records['fold'].append(fold_id)
                    fold_records['image_id'].append(name.replace(' ', '_') + '_%02d'%nb_disease)

            Logger.info('The number of %s | disease: %d, disease image/no disease image/total: %d/%d/%d' % (status,
                                                                                                            disease_counter,dis_img_counter,no_dis_img_counter, total_img))


        df = pd.DataFrame.from_dict(fold_records)
        df.to_csv(args.fold_path, index = False)

        Logger.info('Data preprocessing finished', '-'*20, 'convert file save to %s'%args.input)
    else:
        Logger.info('Data preprocessing folder is exists, load from %s'%args.input)
    return args




def preprocess_raw_data(path, args, preprocess_folder_name = 'preprocess', suffix = '.tif', split_rate = 0.1,seed = 0):
    """
    process the dataset and covert to bounding box based annotation
    Args:
        path:
        args:
        preprocess_folder_name:
        suffix:

    Returns:

    """
    assert osp.exists(path), 'Current dataset path %s is not exists!' %path
    Logger.info('Start to preprocess the dataset in path %s, please wait a moment'%path, '.'*20)
    root_path, data_name = osp.split(path)
    preprocess_path = osp.join(args.voc, '%s_%s'%(preprocess_folder_name, data_name))
    os.makedirs(preprocess_path, exist_ok =True)
    print('Save preprocessing data to %s'%preprocess_path)


    args.num_fold = 0
    args.fold_path = osp.join(preprocess_path, args.mode, 'fold.csv')
    target_path = osp.join(preprocess_path, args.mode)
    if args.mode == 'train' or args.mode == 'eval':
        if args.mode == 'train':
            args.train_xml = osp.join(preprocess_path, args.mode, 'train', 'xml')

        if args.mode == 'eval':
            args.val_xml = osp.join(preprocess_path, args.mode, 'xml')

    args.input = target_path

    if not osp.exists(args.input):
        fold_records = {'fold':[], 'image_id':[]}

        jl = JsonLoader()

        file_path = read_dir(path)
        disease_counter, dis_img_counter, no_dis_img_counter, total_img = 0, 0, 0, 0
        random.seed(seed)
        for name, path in tqdm(file_path.items(), desc='Convert to voc format:'):
            if random.random() > split_rate and args.mode == 'train':
                img_target, mask_target, vis_target, xml_target = mk_dirs(target_path, 'train')
                fold_id = 0
            else:
                img_target, mask_target, vis_target, xml_target = mk_dirs(target_path, 'val')
                fold_id = 1

            jpg_path = '%s.%s'%(path, suffix.strip('.'))
            gt_path = path + '_gt.jpg'
            json_path = path + '.json'


            try:
                assert osp.exists(jpg_path)

                jpg_img = cv2.imread(jpg_path)

                assert len(jpg_img.shape) == 3

                context = jl.load_json(json_path)
                attributes = jl.get_objects(context)
            except:
                print('%s has problemt!' %jpg_path)
                continue



            nb_disease = len(attributes['polygons'])

            disease_counter += nb_disease
            if nb_disease > 0:
                dis_img_counter += 1
            else:
                no_dis_img_counter += 1
            total_img += 1

            if nb_disease > 0:
                # draw bbox image
                # jpg_img_boxes = jl.draw_bboxes(jpg_img, attributes)
                # draw polygon image
                # jpg_img_polygons = jl.draw_polygons(jpg_img, attributes)
                # draw image
                # jpg_mask = jl.draw_mask(jpg_img, attributes, color_dict= {'disease':(1, 1, 1)}, single_channel= True)


                cv2.imwrite(osp.join(img_target, name.replace(' ', '_') + '_%02d.jpg'%nb_disease), jpg_img)
                # cv2.imwrite(osp.join(vis_target, name.replace(' ', '_') + '_%02d.jpg'%nb_disease), cv2.hconcat([jpg_img_boxes, jpg_img_polygons]))
                # cv2.imwrite(osp.join(mask_target, name.replace(' ', '_') + '_%02d.png'%nb_disease), jpg_mask)

                obj_to_xml(osp.join(xml_target, name.replace(' ', '_') + '_%02d.xml'%nb_disease), attributes)


                fold_records['fold'].append(fold_id)
                fold_records['image_id'].append(name.replace(' ', '_') + '_%02d'%nb_disease)

        df = pd.DataFrame.from_dict(fold_records)
        df.to_csv(args.fold_path, index = False)

        Logger.info('The number of disease: %d, disease image/no disease image/total: %d/%d/%d' % (disease_counter,
                                                                                             dis_img_counter,
                                                                                             no_dis_img_counter, total_img))



        Logger.info('Data preprocessing finished', '-'*20, 'covert file save to %s'%args.input)
    else:
        Logger.info('Data preprocessing folder is exists, load from %s'%args.input)
    return args


if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser(description='PyTorch detection')
    # args = parser.parse_args()
    # args.mode = 'train'
    # args = preprocess_raw_data('/home/jovyan/datasets/test_data', args, split_rate=0.5)
    # from project.jbnu.yolov5.data.voc2yolo import voc2yolo_convert
    # voc2yolo_convert(args.input, sets = ['train', 'val'])
    # print(args.input)

    path = '/dataset/khtt/dataset/pine2022/split_test_a'
    path1 = '/dataset/khtt/dataset/pine2022/split_test_b'

    import argparse
    parser = argparse.ArgumentParser(description='PyTorch detection')
    args = parser.parse_args()
    args.voc = '/dataset/khtt/dataset/pine2022/tp'
    args.mode = 'train'
    # args = preprocess_raw_data(path, args, split_rate=0.0, dsize = 800)


    args = preprocess_train_data([path, path1], args, dsize = 800)
    print(args.input)