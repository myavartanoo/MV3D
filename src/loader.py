import shutil
from kitti_data import pykitti
# from kitti_data.pykitti.tracklet import parseXML, TRUNC_IN_IMAGE, TRUNC_TRUNCATED
# from kitti_data.draw import *
from kitti_data.io import *
import net.utility.draw as draw
from net.processing.boxes3d import *
from config import TOP_X_MAX,TOP_X_MIN,TOP_Y_MAX,TOP_Z_MIN,TOP_Z_MAX, \
    TOP_Y_MIN,TOP_X_DIVISION,TOP_Y_DIVISION,TOP_Z_DIVISION
from config import cfg
from tracklets.Tracklet_saver import Tracklet_saver

import os
import cv2
import numpy
import glob
from multiprocessing import Pool
from collections import OrderedDict
import config
import ctypes
from numba import jit
from matplotlib import pyplot as plt
import time
import sys

# raw_dir = cfg.RAW_DATA_SETS_DIR
#
# tracklet_pred_dir = '/home/mohsen/Desktop/MV3D/'
# gt_tracklet = Tracklet_saver(tracklet_pred_dir, 'gt')

remove_list= [0, 2, 7, 8, 10, 12, 13, 14, 16, 19, 22, 23, 24, 25, 32, 33, 39, 40, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 56, 75, 85, 113, 125, 135]
test_list = [item for item in list(range(162)) if item not in remove_list]
def obj_to_gt_boxes3d(objs):

    num        = len(objs)
    gt_boxes3d = np.zeros((num,8,3),dtype=np.float32)
    gt_labels  = np.zeros((num),    dtype=np.int32)

    for n in range(num):
        obj = objs[n]
        b   = obj.box
        label=0
        if obj.type=='Van' or obj.type=='Truck' or obj.type=='Car' or obj.type=='Tram':# todo : only  support 'Van'
            label = 1

        gt_labels [n]=label
        gt_boxes3d[n]=b

    return  gt_boxes3d, gt_labels




text_file_data = open("/home/mohsen/Desktop/MV3D/devkit_object/mapping/train_mapping.txt", "r")
text_file_map = open("/home/mohsen/Desktop/MV3D/devkit_object/mapping/train_rand.txt", "r")



lines = text_file_data.readlines()
list = text_file_map.readlines()
frames_map = list[0].split(',')
#
# count = 0
# for line in lines:
#     l = line.split(' ')
#     data = l[0]
#     drive = l[1][17:21]
#     frames_indexs = [int(l[2])]
#     if data == '2011_09_26':
#         if drive == '0095' or drive =='0101' or drive == '0096' or drive =='0104' or drive =='0106' or drive =='0113' or drive =='0117':
#             print ('salam')
#         else:
#             dataset = pykitti.raw(raw_dir, data, drive, frames_indexs)
#             tracklet_file = os.path.join(dataset.data_path, 'tracklet_labels.xml')
#             objects = read_objects(tracklet_file, frames_indexs)
#             for frames_index in frames_indexs:
#                 if objects != None:
#                     gt_boxes3d, gt_labels = obj_to_gt_boxes3d(objects[0])
#
#                 if len(gt_boxes3d) != 0:
#
#                     gt_translation, gt_size, gt_rotation = boxes3d_decompose(gt_boxes3d[:, :, :])
#
#                     # todo: remove it after gtbox is ok
#                     gt_size[:, 1:3] = gt_size[:, 1:3] / cfg.TRACKLET_GTBOX_LENGTH_SCALE
#
#                     for j in range(len(gt_translation)):
#                         gt_tracklet.add_tracklet(int(frames_map[count]), gt_size[j], gt_translation[j], gt_rotation[j])
#     count = count + 1
# gt_tracklet.write_tracklet()
#

# #################################################################33
# count = 0
# c = 0
# tmp = '0000'
# skip_count = 0
# for line in lines:
#     l = line.split(' ')
#     data = l[0]
#     drive = l[1][17:21]
#     frames_index = int(l[2])
#     if data == '2011_09_26':
#         if drive == '0095' or drive == '0101' or drive == '0096' or drive == '0104' or drive == '0106' or drive == '0113' or drive == '0117' or drive == '0070':
#
#             print('salam')
#         else:
#             #if count%10 == 0:
#                 #print (count)
#                 # src = '/home/mohsen/Desktop/MV3D/data/preprocessing/kitti/gt_boxes3d/' + data + '/' + drive + '/'+l[2][5:-1]+'.npy'
#                 # dst = '/home/mohsen/Desktop/MV3D/data/preprocessed/kitti/gt_boxes3d/'+'object3d/'+'validation/'+frames_map[count]+'.npy'
#                 # shutil.copy(src,dst)
#                 # src = '/home/mohsen/Desktop/MV3D/data/preprocessing/kitti/gt_box_plot/' + data + '/'+ drive + '/'+l[2][5:-1]+'.png'
#                 # dst = '/home/mohsen/Desktop/MV3D/data/preprocessed/kitti/gt_box_plot/'+'object3d/'+'validation/'+frames_map[count]+'.png'
#                 # shutil.copy(src,dst)
#                 # src = '/home/mohsen/Desktop/MV3D/data/preprocessing/kitti/gt_labels/' + data + '/'+ drive + '/'+l[2][5:-1]+'.npy'
#                 # dst = '/home/mohsen/Desktop/MV3D/data/preprocessed/kitti/gt_labels/'+'object3d/'+'validation/'+frames_map[count]+'.npy'
#                 # shutil.copy(src,dst)
#                 # src = '/home/mohsen/Desktop/MV3D/data/preprocessing/kitti/rgb/' + data + '/' + drive + '/'+l[2][5:-1]+'.png'
#                 # dst = '/home/mohsen/Desktop/MV3D/data/preprocessed/kitti/rgb/'+'object3d/'+'validation/'+frames_map[count]+'.png'
#                 # shutil.copy(src,dst)
#                 # src = '/home/mohsen/Desktop/MV3D/data/preprocessing/kitti/top/' + data + '/'+ drive + '/'+l[2][5:-1]+'.npy.npz'
#                 # dst = '/home/mohsen/Desktop/MV3D/data/preprocessed/kitti/top/'+'object3d/'+'validation/'+frames_map[count]+'.npy.npz'
#                 # shutil.copy(src,dst)
#                 # src = '/home/mohsen/Desktop/MV3D/data/preprocessing/kitti/top_image/' + data + '/' + drive + '/'+l[2][5:-1]+'.png'
#                 # dst = '/home/mohsen/Desktop/MV3D/data/preprocessed/kitti/top_image/'+'object3d/'+'validation/'+frames_map[count]+'.png'
#                 # shutil.copy(src,dst)
#             if count%20 == 0:
#                     src = '/home/mohsen/Desktop/MV3D/data/preprocessing/kitti/gt_boxes3d/' + data + '/' + drive + '/'+l[2][5:-1]+'.npy'
#                     dst = '/home/mohsen/Desktop/MV3D/data/preprocessed/kitti/gt_boxes3d/'+'object3d/'+'test/'+tmp+str(c)+'.npy'
#                     shutil.copy(src,dst)
#                     src = '/home/mohsen/Desktop/MV3D/data/preprocessing/kitti/gt_box_plot/' + data + '/'+ drive + '/'+l[2][5:-1]+'.png'
#                     dst = '/home/mohsen/Desktop/MV3D/data/preprocessed/kitti/gt_box_plot/'+'object3d/'+'test/'+tmp+str(c)+'.png'
#                     shutil.copy(src,dst)
#                     src = '/home/mohsen/Desktop/MV3D/data/preprocessing/kitti/gt_labels/' + data + '/'+ drive + '/'+l[2][5:-1]+'.npy'
#                     dst = '/home/mohsen/Desktop/MV3D/data/preprocessed/kitti/gt_labels/'+'object3d/'+'test/'+tmp+str(c)+'.npy'
#                     shutil.copy(src,dst)
#                     src = '/home/mohsen/Desktop/MV3D/data/preprocessing/kitti/rgb/' + data + '/' + drive + '/'+l[2][5:-1]+'.png'
#                     dst = '/home/mohsen/Desktop/MV3D/data/preprocessed/kitti/rgb/'+'object3d/'+'test/'+tmp+str(c)+'.png'
#                     shutil.copy(src,dst)
#                     src = '/home/mohsen/Desktop/MV3D/data/preprocessing/kitti/top/' + data + '/'+ drive + '/'+l[2][5:-1]+'.npy.npz'
#                     dst = '/home/mohsen/Desktop/MV3D/data/preprocessed/kitti/top/'+'object3d/'+'test/'+tmp+str(c)+'.npy.npz'
#                     shutil.copy(src,dst)
#                     src = '/home/mohsen/Desktop/MV3D/data/preprocessing/kitti/top_image/' + data + '/' + drive + '/'+l[2][5:-1]+'.png'
#                     dst = '/home/mohsen/Desktop/MV3D/data/preprocessed/kitti/top_image/'+'object3d/'+'test/'+tmp+str(c)+'.png'
#                     shutil.copy(src,dst)
#
#                     c = c+1
#                     if(c>9):
#                         tmp='000'
#                     if(c>99):
#                         tmp='00'
#                     if(c>999):
#                         tmp='0'
#
#             #else:
#             #    print (count)
#                 # src = '/home/mohsen/Desktop/MV3D/data/preprocessing/kitti/gt_boxes3d/' + data + '/' + drive + '/'+l[2][5:-1]+'.npy'
#                 # dst = '/home/mohsen/Desktop/MV3D/data/preprocessed/kitti/gt_boxes3d/'+'object3d/'+'train/'+frames_map[count]+'.npy'
#                 # shutil.copy(src,dst)
#                 # src = '/home/mohsen/Desktop/MV3D/data/preprocessing/kitti/gt_box_plot/' + data + '/'+ drive + '/'+l[2][5:-1]+'.png'
#                 # dst = '/home/mohsen/Desktop/MV3D/data/preprocessed/kitti/gt_box_plot/'+'object3d/'+'train/'+frames_map[count]+'.png'
#                 # shutil.copy(src,dst)
#                 # src = '/home/mohsen/Desktop/MV3D/data/preprocessing/kitti/gt_labels/' + data + '/'+ drive + '/'+l[2][5:-1]+'.npy'
#                 # dst = '/home/mohsen/Desktop/MV3D/data/preprocessed/kitti/gt_labels/'+'object3d/'+'train/'+frames_map[count]+'.npy'
#                 # shutil.copy(src,dst)
#                 # src = '/home/mohsen/Desktop/MV3D/data/preprocessing/kitti/rgb/' + data + '/' + drive + '/'+l[2][5:-1]+'.png'
#                 # dst = '/home/mohsen/Desktop/MV3D/data/preprocessed/kitti/rgb/'+'object3d/'+'train/'+frames_map[count]+'.png'
#                 # shutil.copy(src,dst)
#                 # src = '/home/mohsen/Desktop/MV3D/data/preprocessing/kitti/top/' + data + '/'+ drive + '/'+l[2][5:-1]+'.npy.npz'
#                 # dst = '/home/mohsen/Desktop/MV3D/data/preprocessed/kitti/top/'+'object3d/'+'train/'+frames_map[count]+'.npy.npz'
#                 # shutil.copy(src,dst)
#                 # src = '/home/mohsen/Desktop/MV3D/data/preprocessing/kitti/top_image/' + data + '/' + drive + '/'+l[2][5:-1]+'.png'
#                 # dst = '/home/mohsen/Desktop/MV3D/data/preprocessed/kitti/top_image/'+'object3d/'+'train/'+frames_map[count]+'.png'
#                 # shutil.copy(src,dst)
#     count = count + 1
#
# #
#
#
# # raw_dir = cfg.RAW_DATA_SETS_DIR
# # dataset = pykitti.raw(raw_dir, '2011_09_26', '0005', [0])
# #
# # tracklet_file = os.path.join(dataset.data_path, 'tracklet_labels.xml')
# # tracklets = parseXML(tracklet_file)
# # print (tracklets[4])
#
#
#


















tmp = '0000'
count = 0
c = 0
for line in lines:
    l = line.split(' ')
    data = l[0]
    drive = l[1][17:21]
    frames_index = int(l[2])
    if data == '2011_09_26':
        if drive == '0095' or drive == '0101' or drive == '0009' or  drive == '0036' or  drive == '0059' or drive == '0096' or drive == '0104' or drive == '0106' or drive == '0113' or drive == '0117' or drive == '0070':

            print('salam')
        else:
                    src = '/home/mohsen/Desktop/MV3D/data/preprocessing/kitti/gt_boxes3d/' + data + '/' + drive + '/'+l[2][5:-1]+'.npy'
                    dst = '/home/mohsen/Desktop/MV3D/data/preprocessed/kitti/gt_boxes3d/'+'object3d/'+'test/'+tmp+str(c)+'.npy'
                    shutil.copy(src,dst)
                    src = '/home/mohsen/Desktop/MV3D/data/preprocessing/kitti/gt_box_plot/' + data + '/'+ drive + '/'+l[2][5:-1]+'.png'
                    dst = '/home/mohsen/Desktop/MV3D/data/preprocessed/kitti/gt_box_plot/'+'object3d/'+'test/'+tmp+str(c)+'.png'
                    shutil.copy(src,dst)
                    src = '/home/mohsen/Desktop/MV3D/data/preprocessing/kitti/gt_labels/' + data + '/'+ drive + '/'+l[2][5:-1]+'.npy'
                    dst = '/home/mohsen/Desktop/MV3D/data/preprocessed/kitti/gt_labels/'+'object3d/'+'test/'+tmp+str(c)+'.npy'
                    shutil.copy(src,dst)
                    src = '/home/mohsen/Desktop/MV3D/data/preprocessing/kitti/rgb/' + data + '/' + drive + '/'+l[2][5:-1]+'.png'
                    dst = '/home/mohsen/Desktop/MV3D/data/preprocessed/kitti/rgb/'+'object3d/'+'test/'+tmp+str(c)+'.png'
                    shutil.copy(src,dst)
                    src = '/home/mohsen/Desktop/MV3D/data/preprocessing/kitti/top/' + data + '/'+ drive + '/'+l[2][5:-1]+'.npy.npz'
                    dst = '/home/mohsen/Desktop/MV3D/data/preprocessed/kitti/top/'+'object3d/'+'test/'+tmp+str(c)+'.npy.npz'
                    shutil.copy(src,dst)
                    src = '/home/mohsen/Desktop/MV3D/data/preprocessing/kitti/top_image/' + data + '/' + drive + '/'+l[2][5:-1]+'.png'
                    dst = '/home/mohsen/Desktop/MV3D/data/preprocessed/kitti/top_image/'+'object3d/'+'test/'+tmp+str(c)+'.png'
                    shutil.copy(src,dst)

                    c = c+1
                    if(c>9):
                        tmp='000'
                    if(c>99):
                        tmp='00'
                    if(c>999):
                        tmp='0'
    count = count + 1

