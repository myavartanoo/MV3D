import mv3d2 as mv3d
import mv3d_net
import glob
from sklearn.utils import shuffle
from config import *
# import utils.batch_loading as ub
import argparse
import os
from utils.training_validation_data_splitter import TrainingValDataSplitter
from utils.batch_loading import BatchLoading2 as BatchLoading


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training')

    all= '%s,%s,%s' % (mv3d_net.top_view_rpn_name ,mv3d_net.imfeature_net_name,mv3d_net.fusion_net_name)

    parser.add_argument('-w', '--weights', type=str, nargs='?', default='',
        help='use pre trained weights example: -w "%s" ' % (all))

    parser.add_argument('-t', '--targets', type=str, nargs='?', default=all,
        help='train targets example: -w "%s" ' % (all))

    parser.add_argument('-i', '--max_iter', type=int, nargs='?', default=1000000,
                        help='max count of train iter')

    parser.add_argument('-n', '--tag', type=str, nargs='?', default='unknown_tag',
                        help='set log tag')

    parser.add_argument('-c', '--continue_train', type=bool, nargs='?', default=False,
                        help='set continue train flag')
    args = parser.parse_args()

    print('\n\n{}\n\n'.format(args))
    tag = args.tag
    if tag == 'unknown_tag':
        tag = input('Enter log tag : ')
        print('\nSet log tag :"%s" ok !!\n' %tag)

    max_iter = args.max_iter
    weights=[]
    if args.weights != '':
        weights = args.weights.split(',')

    targets=[]
    if args.targets != '':
        targets = args.targets.split(',')

    dataset_dir = cfg.PREPROCESSED_DATA_SETS_DIR

    cfg.DATA_SETS_TYPE == 'kitti'
    train_n_val_dataset = [
        # '2011_09_26/2011_09_26_drive_0001_sync', # for tracking
        '2011_09_26/2011_09_26_drive_0002_sync',
        '2011_09_26/2011_09_26_drive_0005_sync',
        # '2011_09_26/2011_09_26_drive_0009_sync',
        '2011_09_26/2011_09_26_drive_0011_sync',
        '2011_09_26/2011_09_26_drive_0013_sync',
        '2011_09_26/2011_09_26_drive_0014_sync',
        '2011_09_26/2011_09_26_drive_0015_sync',
        '2011_09_26/2011_09_26_drive_0017_sync',
        '2011_09_26/2011_09_26_drive_0018_sync',
        '2011_09_26/2011_09_26_drive_0019_sync',
        '2011_09_26/2011_09_26_drive_0020_sync',
        '2011_09_26/2011_09_26_drive_0022_sync',
        '2011_09_26/2011_09_26_drive_0023_sync',
        '2011_09_26/2011_09_26_drive_0027_sync',
        '2011_09_26/2011_09_26_drive_0028_sync',
        '2011_09_26/2011_09_26_drive_0029_sync',
        '2011_09_26/2011_09_26_drive_0032_sync',
        '2011_09_26/2011_09_26_drive_0035_sync',
        '2011_09_26/2011_09_26_drive_0036_sync',
        '2011_09_26/2011_09_26_drive_0039_sync',
        '2011_09_26/2011_09_26_drive_0046_sync',
        '2011_09_26/2011_09_26_drive_0048_sync',
        '2011_09_26/2011_09_26_drive_0051_sync',
        '2011_09_26/2011_09_26_drive_0052_sync',
        '2011_09_26/2011_09_26_drive_0056_sync',
        '2011_09_26/2011_09_26_drive_0057_sync',
        '2011_09_26/2011_09_26_drive_0059_sync',
        '2011_09_26/2011_09_26_drive_0060_sync',
        '2011_09_26/2011_09_26_drive_0061_sync',
        '2011_09_26/2011_09_26_drive_0064_sync',
        '2011_09_26/2011_09_26_drive_0070_sync',
        '2011_09_26/2011_09_26_drive_0079_sync',
        '2011_09_26/2011_09_26_drive_0084_sync',
        '2011_09_26/2011_09_26_drive_0086_sync',
        '2011_09_26/2011_09_26_drive_0087_sync',
        '2011_09_26/2011_09_26_drive_0091_sync',
        # '2011_09_26/2011_09_26_drive_0093_sync',  #data size not same
        # '2011_09_26/2011_09_26_drive_0095_sync',
        # '2011_09_26/2011_09_26_drive_0096_sync',
        # '2011_09_26/2011_09_26_drive_0104_sync',
        # '2011_09_26/2011_09_26_drive_0106_sync',
        # '2011_09_26/2011_09_26_drive_0113_sync',
        # '2011_09_26/2011_09_26_drive_0117_sync',
        '2011_09_26/2011_09_26_drive_0119_sync',
    ]

    # shuffle bag list or same kind of bags will only be in training or validation set.
train_n_val_dataset = shuffle(train_n_val_dataset, random_state=666)
data_splitter = TrainingValDataSplitter(train_n_val_dataset)

with BatchLoading(tags=data_splitter.training_tags, require_shuffle=True, random_num=np.random.randint(100),
                  is_flip=False) as training:
    with BatchLoading(tags=data_splitter.val_tags, queue_size=1, require_shuffle=True, random_num=666) as validation:
        train = mv3d.Trainer(train_set=training, validation_set=validation,
                             pre_trained_weights=weights, train_targets=targets, log_tag=tag,
                             continue_train=args.continue_train)

        train(max_iter=max_iter)