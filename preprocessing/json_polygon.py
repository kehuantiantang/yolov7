# coding=utf-8
import json
import sys
from types import SimpleNamespace
import os
import os.path as osp
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm


class JsonLoader(object):
    color_dict = {'red':(0, 0, 255), 'green':(0, 255, 0), 'blue':(255, 0,
                                                                  0),
                  'yellow': (255, 255, 0), 'cyan':(0, 255, 255), 'sliver':(
            192, 192, 192), 'black': (255, 255, 255)}

    def __init__(self, get_object_method = None):
        self.get_object_method = get_object_method

    def get_color(self, c):
        if isinstance(c, str):
            return JsonLoader.color_dict[c]
        else:
            return c

    def load_json(self, path, replace_pair = [('class', 'class_id'), ('Class', 'Class_id')], encoding = 'utf-8',
                  skip = 0,):
        with open(path, encoding=encoding) as f:
            context = f.read()[skip:]
            if replace_pair:
                for old, new in replace_pair:
                    context = context.replace(old, new)

            context = json.loads(context, object_hook=lambda d: SimpleNamespace(**d))
            return context

    def get_objects(self, context, keep_shape_type = ['polygon']):
        if self.get_object_method != None:
            return self.get_object_method(context)
        else:
            width =  int(context.imageWidth)
            height = int(context.imageHeight)
            path = context.imagePath
            obj_dicts = {'name':[], 'bboxes':[], 'category_name':[],
                         'name_pattern': '', 'height':height, 'width':width, 'path':path, 'polygons':[],
                         'filename':path}

            for polygon in context.shapes:
                label = polygon.label
                points = polygon.points
                group_id = polygon.group_id

                shape_type = polygon.shape_type

                if shape_type in keep_shape_type:

                    flags = polygon.flags


                    x_point, y_point = [], []
                    for point in points:
                        h, w = point
                        x_point.append(h)
                        y_point.append(w)

                    if len(x_point) > 0 and len(y_point) >0:
                        # avoid the out of range
                        xmin, ymin, xmax, ymax = max(min(x_point), 0), max(min(y_point), 0), min(max(x_point), width), min(max(y_point),
                                                                                                         height)
                        if xmin >= xmax or ymin >= ymax:
                            continue

                        if shape_type == 'polygon':
                            obj_dicts['name'].append('disease')
                            obj_dicts['bboxes'].append([xmin, ymin, xmax, ymax])
                            obj_dicts['category_name'].append('disease')
                            obj_dicts['polygons'].append(points)



            return obj_dicts


    def draw_box(self, img, bb, name, c):
        color = self.get_color(c)
        bb = [int(float(b)) for b in bb]
        img = cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), color, 1)

        text_pos = [bb[0], bb[1] - 10]
        # 'xmin', 'ymin', 'xmax', 'ymax'
        if text_pos[1] < 0:
            text_pos = [bb[2], bb[3] - 10]
        img = cv2.putText(img, str(name), tuple(text_pos),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, JsonLoader.color_dict["green"], 1, cv2.LINE_AA)
        return img

    def draw_bboxes(self, img, obj_dicts):
        assert np.max(img) > 1.0
        img = np.array(img)
        for box, name in zip(obj_dicts['bboxes'], obj_dicts['name']):
            img = self.draw_box(img, box, name, 'blue')
        return img


    def draw_polygon(self, img, polygon, name, c):
        color = self.get_color(c)
        img = cv2.polylines(img, [np.array(polygon, np.int32)], True, color, 1)
        cv2.putText(img, str(name), (max(0, int(polygon[0][0])), max(10, int(polygon[0][1] - 10))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, JsonLoader.color_dict["green"], 1, cv2.LINE_AA)
        return img

    def draw_polygons(self, img, obj_dicts, color_dict = None):
        assert np.max(img) > 1.0
        img = np.array(img)
        try:
            if color_dict is None:
                color_dict = {obj_dicts['name'][0] : 'blue'}
            for polygon, name in zip(obj_dicts['polygons'], obj_dicts['name']):
                img = self.draw_polygon(img, polygon, name, color_dict[name])
        except:
            print(obj_dicts)
        return img

    def draw_mask(self, img, obj_dicts, color_dict = None, single_channel = False):
        '''
        draw semantic segmentation mask
        :param img:  raw gis image
        :param obj_dicts:  polygon point
        :param color_dict:  {object_name: (0, 0, 0)} object --> color dict
        :param single_channel: if true return segmentation mask [0, 1, 0, 0, 2, 1],
        else return rgb visualization mask
        :return:
        '''
        mask = np.zeros_like(img)
        obj_polygons = defaultdict(list)
        for polygon, name in zip(obj_dicts['polygons'], obj_dicts['name']):
            obj_polygons[name].append(np.array(polygon).astype(np.int32))


        for name, polygons in obj_polygons.items():
            mask = cv2.fillPoly(mask, polygons, color_dict[name] if color_dict is not None else self.get_color('black'))


        if single_channel:
            mask = mask[:, :, 0]
            return mask
        else:
            return mask


    def object_counter_bbox_polygon(self, path):
        counter, bboxes, polygons = defaultdict(int), defaultdict(list), defaultdict(list)

        for root, _, filenames in os.walk(path):
            for filename in tqdm(filenames):
                if filename.endswith('json'):
                    p = osp.join(root, filename)

                    context = JsonLoader.load_json(p)
                    obj_dicts = JsonLoader.get_objects(context)

                    for cls, bbox, polygon in zip(obj_dicts['name'], obj_dicts['bboxes'], obj_dicts['polygons']):
                        counter[cls] += 1
                        bboxes[cls].append(bboxes)
                        polygons[cls].append(polygon)

        return counter, bboxes, polygons

