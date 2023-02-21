import xml.etree.ElementTree as ET
import os
import os.path as osp

classes = ['disease']


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(source, image_id, target = None):
    in_file = open(osp.join(source, '%s.xml' % (image_id)))
    if target is None:
        target = source
    out_file = open(osp.join(target, '%s.txt' % (image_id)), 'w')

    tree = ET.parse(in_file)
    source = tree.getroot()
    size = source.find('size')
    w = int(float(size.find('width').text))
    h = int(float(size.find('height').text))

    for obj in source.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def voc2yolo_convert(root = '/home/jovyan/datasets/yolo', sets = ['train', 'val']):

    for image_set in sets:
        if not os.path.exists(root):
            os.makedirs(root)

        image_ids = [f.split('.')[0] for f in os.listdir(osp.join(root, image_set, 'img')) if f.endswith('jpg')]

        list_file = open(os.path.join(root, '%s.txt'%(image_set)), 'w')
        for image_id in image_ids:
            string = osp.join(root, image_set, 'img', '%s.jpg' % image_id) + '\n'
            list_file.write(string)
            convert_annotation(osp.join(root, image_set, 'xml'), image_id, target = osp.join(root, image_set, 'img'))
        list_file.close()