# coding=utf-8
# @Project  ：fasterrcnn_deploy 
# @FileName ：json_utils.py
# @Author   ：SoberReflection
# @Revision : sober 
# @Date     ：2022/12/11 2:08 下午
import copy
import json
import os
import os.path as osp
import warnings

import cv2
import numpy as np


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

def draw_bboxes(img, obj_dicts, color = (255, 0, 0)):
    assert np.max(img) > 1.0
    img = np.array(img)
    for box, name in zip(obj_dicts['bboxes'], obj_dicts['name']):
        img = draw_box(img, box, name, color)
    return img


class JsonTemplate(object):
    def __init__(self, name, raw_height=-1, raw_width=-1):
        self.raw_height, self.raw_width, self.target_height, self.target_width = raw_height, raw_width, -1, -1
        self.template = {"version": "4.5.7", "flags": {}, "shapes": [], "imagePath": "", "imageData": "",
            "imageHeight": self.target_height, "imageWidth": self.target_width}
        self.shape_template = {"label": "00000000", 'score': -1, "points": [], "group_id": None,
            "shape_type": "rectangle", "flags": {}}
        self.is_set_height_width = False
        self.pred_bboxes, self.gt_bboxes = [], []
        self.name = name
        self.save_json_path, self.save_img_path = None, None
        self.raw_image_path = None

    def set_raw_image_path(self, root_path, extension):
        self.raw_image_path = osp.join(root_path, '%s.%s' % (self.name, extension))
        assert osp.exists(
            self.raw_image_path), '%s is not exist, please delete preprocess folder and rebuild' % self.raw_image_path

    def __rescale_bboxes(self, bboxes):
        bboxes = np.array(bboxes).reshape((-1, 5))
        bboxes[:, [1, 3]], bboxes[:, [2, 4]] = bboxes[:, [1, 3]] / self.raw_width * self.target_width, bboxes[:, [2,
                                                                                                                  4]] / self.raw_height * self.target_height
        return bboxes

    def draw_pred_gt_bboxes(self, path = None, conf_threshold = -1):
        if osp.exists(self.raw_image_path):
            img = cv2.imread(self.raw_image_path)
            height, width, _ = img.shape
            self.set_height_width(height, width)
            # gt
            if len(self.gt_bboxes) > 0:
                gt_bboxes = self.__rescale_bboxes(self.gt_bboxes)
                img = draw_bboxes(img, {'bboxes': gt_bboxes[:, 1:], 'name': ['' for _ in range(len(gt_bboxes))]},
                                  color=(255, 0, 0))

            # pred
            if len(self.pred_bboxes) > 0:
                pred_bboxes = self.__rescale_bboxes(self.pred_bboxes)
                pred_bboxes = pred_bboxes[pred_bboxes[:, 0] >= conf_threshold]
                img = draw_bboxes(img, {'bboxes': pred_bboxes[:, 1:], 'name': ['%.4f'%v for v in pred_bboxes[:, 0]]},
                              color=(0, 255, 0))

            if path == None:
                assert self.save_img_path is not None
                path = self.save_img_path

            cv2.imwrite(osp.join(path, '%s.jpg' % self.name), img)
        else:
            warnings.warn("Image is not exist !, %s" % self.raw_image_path)

    def state_of_height_width(self):
        return self.is_set_height_width

    def set_height_width(self, height, width):
        self.target_height = height
        self.target_width = width
        self.is_set_height_width = True

    def add_bbox(self, score, xmin, ymin, xmax, ymax, status):

        if status == 'pred':
            self.pred_bboxes.append([score, xmin, ymin, xmax, ymax])
        elif status == 'gt':
            self.gt_bboxes.append([score, xmin, ymin, xmax, ymax])
        else:
            raise ValueError("Status value must be [pred, gt]")


    def add_bboxes(self, scores, bboxes, status):
        if status == 'gt':
            scores = [2.0 for _ in range(len(bboxes))]
        for score, (xmin, ymin, xmax, ymax) in zip(scores, bboxes):
            self.add_bbox(score, xmin, ymin, xmax, ymax, status)

    def add_list_bboxes(self, list_bboxes, status):
        '''
        add bbox that arrage with [score, xmin, ymin, xmax, ymax]
        '''
        if len(list_bboxes) > 0:
            if status == 'gt':
                self.add_bboxes(None, list_bboxes, status)
            else:
                list_bboxes = np.array(list_bboxes)
                self.add_bboxes(list_bboxes[:, 0], list_bboxes[:, 1:], status)

    def set_save_json_path(self, path):
        if self.save_json_path is None:
            os.makedirs(path, exist_ok=True)
            self.save_json_path = path

    def set_save_img_path(self, path):
        if self.save_img_path is None:
            os.makedirs(path, exist_ok=True)
            self.save_img_path = path

    def __construct_content(self, conf_threhsold):
        pred_boxes = self.__rescale_bboxes(self.pred_bboxes)
        for score, xmin, ymin, xmax, ymax in pred_boxes:
            if score >= conf_threhsold:
                shape_template = copy.deepcopy(self.shape_template)
                shape_template['score'] = "%.2f" % (float(score) * 100)
                shape_template['points'] = [[round(xmin), round(ymin)], [round(xmax), round(ymax)]]
                self.template['shapes'].append(shape_template)
        self.template['imageHeight'], self.template['imageWidth'] = self.target_height, self.target_width

    def write(self, path=None, conf_threhsold=-1):
        self.__construct_content(conf_threhsold)
        if path == None:
            assert self.save_json_path is not None
            path = self.save_json_path

        with open(osp.join(path, '%s.json' % self.name), 'w', encoding='utf-8') as f:
            json.dump(self.template, f, indent=6)
