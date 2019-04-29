#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os,sys, glob, re, shutil
import scipy.misc

FILE_LOCATION = os.path.dirname(os.path.abspath(__file__))
import pycococreatortools as pct
from pycococreatortools import resize_binary_mask as rbm

ROOT_DIR = os.path.join('..', 'data', 'train')
IMAGE_DIR = os.path.join(ROOT_DIR, "fs_train_tmp")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

ANNOTATION_DIR = os.path.join(ROOT_DIR, "annotations_tmp")
if not os.path.exists(ANNOTATION_DIR):
    os.makedirs(ANNOTATION_DIR)

def getBBoxes(xml_file):
    fh = open(xml_file)
    data = []

    for line in fh:
        if re.search('Header|Table|xmin|xmax|ymin|ymax',line):
            m = re.search('>(.+?)<', line)
            if m:
                ext_data = m.group(1)

                if re.search('name', line):
                    btype = ext_data
                elif re.search('xmin', line):
                    xmin = int(ext_data)
                elif re.search('ymin', line):
                    ymin = int(ext_data)
                elif re.search('xmax', line):
                    xmax = int(ext_data)
                elif re.search('ymax', line):
                    ymax = int(ext_data)

                    b_data = [btype, xmin, xmax, ymin, ymax]

                    data.append(b_data)

    return data


def genData(data_dir):
    files  = glob.glob(os.path.join(data_dir,'*.xml'))

    # data = []
    for f in files:
        print('Processing:', f)
        # img_path_name = os.path.splitext(f)[0]
        img_name = os.path.basename(f)
        img_name = img_name.replace('.xml','')
        img_file = os.path.splitext(f)[0]+'.jpg'
        if os.path.exists(img_file):
            f_data = []
            print(img_file)
            img_np = np.array(Image.open(img_file))
            W, H = img_np.shape[:2]
            f_data.append([W,H])

            bbox_data = getBBoxes(f)
            # if len(bbox_data) > 0:
            #     f_data.append(bbox_data)
            #     data.append(f_data)
            #     print('Generating Mask', W, H, bbox_data)
            #     mask = genMask(W, H, bbox_data)

            #     _, ax = plt.subplots(nrows=1, ncols=2)
            #     ax[0].imshow(img_np)
            #     ax[1].imshow(mask)
            #     plt.show()

            shutil.copy(img_file, IMAGE_DIR)

            for bbox in bbox_data:
                print('Generating Mask', W, H, bbox)
                mask = genMask(W, H, bbox)
                # Copying data
                # print('debug:',bbox[0][:len(bbox[0]-1)])
                m_name = img_name + '_' + bbox[0][:len(bbox[0])-1] + '_' + bbox[0][len(bbox[0])-1] + '.jpg'
                m_path = os.path.join(ANNOTATION_DIR, m_name)
                print('Saving file:', m_path)
                scipy.misc.imsave(m_path, mask)

        else:
            continue

def genMask(W, H, bbox):

    # Generate the image mask
    mask = np.zeros((W,H))
    print('region:', mask.shape)

    print('bbox:',bbox)
    _, xmin, xmax, ymin, ymax = bbox
    mask[ymin:ymax,xmin:xmax] = 1

    return mask

def genMasks(W, H, bboxes):

    # Generate the image mask
    mask = np.ones((W,H))
    print('region:', mask.shape)

    for bbox in bboxes:
        print('bbox:',bbox)
        _, xmin, xmax, ymin, ymax = bbox
        mask[ymin:ymax,xmin:xmax] = 0

    return mask

def main():
    data_dir = os.path.join(FILE_LOCATION, '..', 'data', 'images_xmls_tmp')
    print(data_dir)

    genData(data_dir)

    pass

if __name__ == '__main__':
    main()
