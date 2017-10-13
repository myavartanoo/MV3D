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

    parser.add_argument('-i', '--max_iter', type=int, nargs='?', default=100000,
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


    if cfg.DATA_SETS_TYPE == 'kitti':
        train_n_val_dataset = [
            # '2011_09_26/2011_09_26_drive_0001_sync', # for tracking
            '2011_09_26/2011_09_26_drive_0001_sync',
         ]

        validation_dataset = {
            '2011_09_26': ['0001','0002','0005','0011','0013','0014','0017','0018','0048',
                           '0051','0056','0057','0059','0060','0084','0091','0093','0095',
                           '0096','0104','0106','0113','0117']}

        train_n_val_dataset = shuffle(train_n_val_dataset, random_state=666)
        data_splitter = TrainingValDataSplitter(train_n_val_dataset)

        with BatchLoading(tags=data_splitter.training_tags, require_shuffle=True, random_num=np.random.randint(100),
                          is_flip=False) as training:
            with BatchLoading(tags=data_splitter.val_tags, queue_size=1, require_shuffle=True,
                              random_num=666) as validation:
                #train = mv3d.Trainer(train_set=training, validation_set=validation,
                                     #pre_trained_weights=weights, train_targets=targets, log_tag=tag,
                                     #continue_train=args.continue_train,
                                     #fast_test_mode=True if max_iter == 1 else False)
                train = mv3d.Trainer(train_set=training, validation_set=validation,
                                     pre_trained_weights=weights, train_targets=targets, log_tag=tag,
                                     continue_train=args.continue_train)
                train(max_iter=max_iter)