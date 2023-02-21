# coding=utf-8
from collections import defaultdict
import os
from jinja2 import Environment, PackageLoader, FileSystemLoader
from xml.etree.ElementTree import parse


class Writer(object):
    def __init__(self, path, width, height, depth=3, database='Unknown', segmented=0):
        # environment = Environment(loader=PackageLoader('./', 'resource'),
        #                           keep_trailing_newline=True)
        environment = Environment(loader=FileSystemLoader(searchpath="resource"), keep_trailing_newline=True)
        self.annotation_template = environment.get_template('annotation.xml')

        abspath = os.path.abspath(path)

        self.template_parameters = {
            'path': abspath,
            'filename': os.path.basename(abspath),
            'folder': os.path.basename(os.path.dirname(abspath)),
            'width': width,
            'height': height,
            'depth': depth,
            'database': database,
            'segmented': segmented,
            'objects': []
        }

    def addObject(self, name, xmin, ymin, xmax, ymax, pose='Unspecified', truncated=0, difficult=0):
        self.template_parameters['objects'].append({
            'name': name,
            'xmin': round(float(xmin)),
            'ymin': round(float(ymin)),
            'xmax': round(float(xmax)),
            'ymax': round(float(ymax)),
            'pose': pose,
            'truncated': truncated,
            'difficult': difficult,
        })

    def addObjectByBox(self, bbox, name):
        xmin, ymin, xmax, ymax = [round(float(b)) for b in bbox]
        self.template_parameters['objects'].append({
            'name': name,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'pose': 'Unspecified',
            'truncated': 0,
            'difficult': 0,
        })

    def addBboxes(self, bboxes, names, difficults= None):
        if not isinstance(names, list):
            names = [names for _ in range(len(bboxes))]
        if difficults is None:
            difficults = [0 for _ in range(len(names))]
        for box, name, difficult in zip(bboxes, names, difficults):
            xmin, ymin, xmax, ymax = map(round, map(float, box))
            self.addObject(name, xmin, ymin, xmax, ymax, difficult = difficult)


    def save(self, annotation_path):
        if not annotation_path.endswith('.xml'):
            annotation_path = annotation_path.split('.')[0] + '.xml'

        with open(annotation_path, 'w', encoding= 'utf-8') as file:
            content = self.annotation_template.render(**self.template_parameters)
            file.write(content.encode('utf-8').decode('utf-8'))


class Reader(object):
    def __init__(self, xml_path):
        self.xml_path = xml_path
        # assert os.path.exists(self.xml_path)
        if os.path.exists(self.xml_path):
            tree = parse(xml_path)
            self.root = tree.getroot()
        else:
            self.root = None

        self.name_pattern = os.path.split(xml_path)[-1].split('_')[-1].split(
            '.')[0]

    def text2int(self, text):
        return int(round(float(text)))

    def get_objects(self):
        # objects = self.root.getElementsByTagName("object")
        width = float(self.root.find("size").find('width').text)
        height = float(self.root.find("size").find('height').text)
        path = self.root.find("path").text
        obj_dicts = {'name':[], 'bboxes':[], 'category_name':[], 'difficult':[],
                     'name_pattern': '', 'height':height, 'width':width, 'path':path}

        if self.root is not None:
            for object in self.root.iter('object'):
                difficult = int(object.find('difficult').text)

                box = object.find('bndbox')
                y_min = self.text2int(box.find("ymin").text)
                x_min = self.text2int(box.find("xmin").text)
                y_max = self.text2int(box.find("ymax").text)
                x_max = self.text2int(box.find("xmax").text)
                bbox = [x_min, y_min, x_max, y_max]

                name = object.find('name').text
                obj_dicts['name'].append(name)
                obj_dicts['category_name'].append(name)
                obj_dicts['bboxes'].append(bbox)
                obj_dicts['difficult'].append(difficult)

                obj_dicts['name_pattern'] = self.name_pattern
        return obj_dicts

    def get_objectByName(self, name):
        objects = self.root.getElementsByTagName("object")
        obj_dicts = defaultdict(list)

        for object in objects:
            box = object.find('bndbox')

            y_min = int(box.find("ymin").text)
            x_min = int(box.find("xmin").text)
            y_max = int(box.find("ymax").text)
            x_max = int(box.find("xmax").text)
            bbox = [x_min, y_min, x_max, y_max]

            current_name = object.find('name').text
            if current_name == name:
                obj_dicts['category_name'].append(current_name)
                obj_dicts['bboxes'].append(bbox)
                obj_dicts['name_pattern'].append(self.name_pattern)
        return obj_dicts