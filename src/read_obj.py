import numpy as np




def computeBox3D(object, P):

    face_idx = np.array([[1, 2, 6, 5],\
                [2, 3, 7, 6],\
                [3, 4, 8, 7],\
                [4, 1, 5, 8]])

    # R = np.array([ \
    #     [np.cos(object.ry), 0.0, np.sin(object.ry)], \
    #     [-np.sin(object.ry), 0.0, np.cos(object.ry)],\
    #     [0.0, 1.0, 0.0]
    #     ])

    R =np.array([ \
        [np.cos(object.ry), -np.sin(object.ry), 0.0], \
        [np.sin(object.ry), np.cos(object.ry), 0.0], \
        [0.0, 0.0, 1.0]])
    #print (R)

    l = object.l
    w = object.w
    h = object.h

    corners = np.array([  # in velodyne coordinates around zero point and without orientation yet\
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], \
        [w/2, -w/ 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],\
        [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]])
    #
    # corners =   np.array([[-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
    #     [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
    #     [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2]])


    corners_3D       = np.dot(R, corners)
    corners_3DD      = np.zeros([3,8])
    corners_3DD[0,:] = corners_3D[1,:] + object.t3
    corners_3DD[1,:] = corners_3D[0,:] - object.t1
    corners_3DD[2,:] = corners_3D[2,:] - object.t2 +0.81

    if any(corners_3DD[2,:]<0.1):
        corners_2D = []
    # print('h')
    # print(h)
    # print('l')
    # print(l)
    # print('w')
    # print(w)
    # print(object.t1)
    # print(object.t2)
    # print(object.t3)
    #
    # print(corners_3DD.transpose())
    corners_2D = projectToImage(corners_3D,P)
    gt_label = 0
    if object.type == 'Car' or object.type == 'Van':
        gt_label = 1
    print (P)
    return corners_3DD.transpose(), gt_label, P, R




def computeBox3D_Inverse(boxes):

    objects = []
    for box in boxes:
        MyObject = type('MyObject', (object,), {})
        obj = MyObject()
        obj.h           = abs(box[0,2]-box[4,2])
        obj.w           = abs(box[0,0]-box[2,0])
        obj.l           = abs(box[1,1]-box[2,1])
        obj.x1          =
        obj.y1          =
        obj.x2          =
        obj.y2          =
        obj.t1          =
        obj.t2          =
        obj.t3          =
        obj.ry          =
        obj.occlusion   = '3'
        obj.truncation  =
        obj.alpha       =
        obj.type        = 'Car'
        obj.socre       =















def readLabels(label_dir,img_idx):

    fid     = open(label_dir+'/%06d.txt'%img_idx, "r")
    lists   = []
    lines   = fid.readlines()
    for line in lines:
        list = line.split()
        lists.append(list)
    objects = []
    for o in range(len(lines)):
      MyObject = type('MyObject', (object,), {})
      obj      = MyObject()


      obj.type       = lists[o][0]
      obj.truncation = np.float(lists[o][1])
      obj.occlusion  = np.float(lists[o][2])
      obj.alpha      = np.float(lists[o][3])
      obj.x1         = np.float(lists[o][4])
      obj.y1         = np.float(lists[o][5])
      obj.x2         = np.float(lists[o][6])
      obj.y2         = np.float(lists[o][7])
      obj.h          = np.float(lists[o][8])
      obj.w          = np.float(lists[o][9])
      obj.l          = np.float(lists[o][10])
      obj.t1         = np.float(lists[o][11])
      obj.t2         = np.float(lists[o][12])
      obj.t3         = np.float(lists[o][13])
      obj.ry         = np.float(lists[o][14])
      if obj.type == 'Car':
        objects.append(obj)

    return objects


def readCalibration(calib_dir, img_idx, cam):

    fid                  = open(calib_dir + '/%06d.txt' % img_idx, "r")
    lines                = fid.readlines()
    list_P               = lines[cam].split()
    P_arr                = list_P[1:]
    P_arr                = [float(i) for i in P_arr]
    P                    = np.reshape(P_arr, [4, 3]).transpose()
    list_R0_rect         = lines[cam+2].split()
    R0_rect_arr          = list_R0_rect[1:]
    R0_rect_arr          = [float(i) for i in R0_rect_arr]
    R0_rect              = np.reshape(R0_rect_arr, [3, 3])
    list_Tr_velo_to_cam  = lines[cam+3].split()
    Tr_velo_to_cam_arr   = list_Tr_velo_to_cam[1:]
    Tr_velo_to_cam_arr   = [float(i) for i in Tr_velo_to_cam_arr]
    Tr_velo_to_cam       = np.reshape(Tr_velo_to_cam_arr, [3, 4])




    return P, R0_rect, Tr_velo_to_cam




def projectToImage(pts_3D, P):
  a           = np.array(pts_3D)
  b           = np.array([np.ones(np.size(a,1))])
  pts_2D      = np.dot(P , np.concatenate((a,b),axis=0))
  pts_2D[0,:] = np.divide(pts_2D[0,:],pts_2D[2,:])
  pts_2D[1,:] = np.divide(pts_2D[1,:],pts_2D[2,:])


  return np.delete(pts_2D,2,0)



def   computeOrientation3D(object,P):


    R = np.array([ \
        [np.cos(object.ry), -np.sin(object.ry), 0.0], \
        [np.sin(object.ry), np.cos(object.ry), 0.0], \
        [0.0, 0.0, 1.0]])

    orientation_3D = np.array([[0.0, object.l],\
                     [0.0, 0.0],\
                     [ 0.0, 0.0]])

    orientation_3D      = np.dot(R, orientation_3D)
    orientation_3D[0,:] = orientation_3D[0,:] + object.t1
    orientation_3D[1,:] = orientation_3D[1,:] + object.t2
    orientation_3D[2,:] = orientation_3D[2,:] + object.t3

    if any(orientation_3D[2,:]<0.1):
        orientation_2D = []
    return R

    orientation_2D = projectToImage(orientation_3D,P)


# def drawBox3D(h, object, corners, face_idx, orientation):
#
#
#     occ_col = ['g', 'y', 'r', 'w']
#     trun_style = ['-', '--']
#     trc = np.float(object.truncation > 0.1) + 1;
#
#
#     if corners is []:
#         for f in range(3)
#             cv2.line(h, (corners(0,face_idx(f,:)),corners(0,face_idx(f,0))), (corners(1,face_idx(f,:)),corners(1,face_idx(f,0))), (0,0,255), thickness, cv2.LINE_AA)
#




objects = readLabels('/home/mohsen/Desktop/MV3D/data/raw/kitti/object3d_all/object3d_all_drive_all_sync/label_2',4736)
P, R0_rect, Tr_velo_to_cam = readCalibration('/home/mohsen/Desktop/MV3D/data/raw/kitti/object3d_all/object3d_all_drive_all_sync/calib',4736,2)

computeBox3D(objects[0],P)