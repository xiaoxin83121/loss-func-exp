from __future__ import absolute_import
from matplotlib import pyplot as plt
import os
import xml.etree.ElementTree as ET
import json
import numpy as np

voc_path = "E:\\resource\\VOC\\VOCdevkit\\VOC2012"
voc_anno_path = "Annotations"


def read_xml(filename):
    tree = ET.ElementTree(file=filename)
    root = tree.getroot()
    objs = root.findall('object')
    res = []
    for obj in objs:
        bndbox = obj.find('bndbox')
        xmin, ymin, xmax, ymax = float(bndbox.find('xmin').text), float(bndbox.find('ymin').text), \
                                 float(bndbox.find('xmax').text), float(bndbox.find('ymax').text)
        label = obj.find('name').text
        res.append({'label': label, 'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax,
                    'width': xmax - xmin, 'height': ymax - ymin,
                    'aspect_ratio': (xmax - xmin)/(ymax - ymin)})
    return res


def dump_json(voc_anno_path):
    files = os.listdir(voc_anno_path)
    cls_bboxes = {}
    for filename in files:
        file_path = os.path.join(voc_anno_path, filename)
        bboxes = read_xml(file_path)
        for bbox in bboxes:
            label = bbox.get('label')
            if label in cls_bboxes:
                cls_bboxes.get(label).append(bbox)
            else:
                cls_bboxes[label] = [bbox]
    # cls_bboxes = sorted(cls_bboxes.keys())
    for cls in sorted(cls_bboxes.keys()):
        print("{}={}".format(cls, len(cls_bboxes[cls])))
    writer = open('bbox.json', 'w')
    json.dump(cls_bboxes, writer)


def cal_aspect_ratios(filename):
    reader = open(filename, 'r')
    cls_bboxes = json.load(reader)
    if not os.path.exists('aspect_ratio.json'):
        cls_aspect_ratio = {}
        for cls in sorted(cls_bboxes.keys()):
            bboxes = cls_bboxes[cls]
            aspect_ratios = [round(bbox['aspect_ratio'], 2) for bbox in bboxes]
            cls_aspect_ratio[cls] = aspect_ratios
        writer = open('aspect_ratio.json', 'w')
        json.dump(cls_aspect_ratio, writer)
        writer.close()
    else:
        reader = open('aspect_ratio.json', 'r')
        cls_aspect_ratio = json.load(reader)
        reader.close()

    for cls in sorted(cls_aspect_ratio.keys())[:1]:
        plt.title(cls)
        x = np.arange(0, max(cls_aspect_ratio[cls]), step=0.01)
        y = np.zeros_like(x)
        for aspect_ratio in cls_aspect_ratio[cls]:
            y[np.where(x==aspect_ratio)[0]] += 1
        # plt.subplot()
        plt.plot(x, y)
        plt.show()




if __name__ == "__main__":
    voc_anno_path = os.path.join(voc_path, voc_anno_path)
    # files = os.listdir(voc_anno_path)
    # read_xml(os.path.join(voc_anno_path, files[0]))
    # dump_json(voc_anno_path)
    cal_aspect_ratios('bbox.json')
