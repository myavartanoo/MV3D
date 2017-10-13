import mv3d2 as mv3d
import mv3d_net
from train import *
from tracking import *
from data import *
import glob
from sklearn.utils import shuffle
from config import *
# import utils.batch_loading as ub
import argparse
import os
from utils.training_validation_data_splitter import TrainingValDataSplitter
from utils.batch_loading import BatchLoading2 as BatchLoading
from tracking import str2bool
import time

if __name__ == '__main__':
    all = '%s,%s,%s' % (mv3d_net.top_view_rpn_name, mv3d_net.imfeature_net_name, mv3d_net.fusion_net_name)

    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("-test", help="running test", action="store_true")
    parser.add_argument("-tr", "--training", help="running trainer", action="store_true")
    parser.add_argument("-data", help="running data", action="store_true")
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
    parser.add_argument('-f', '--fast_test', type=str2bool, nargs='?', default=False,
                        help='set fast_test model')
    parser.add_argument('-s', '--n_skip_frames', type=int, nargs='?', default=0,
                        help='set number of skip frames')
    args = parser.parse_args()



    if args.training:
        print ('Ok! you runned the training')

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
                '2011_09_26': ['0051']
            }

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


    elif args.test:
        print ('Ok! you runned the testing')

        print('\n\n{}\n\n'.format(args))
        tag = args.tag
        if tag == 'unknown_tag':
            tag = input('Enter log tag : ')
            print('\nSet log tag :"%s" ok !!\n' % tag)
        weights_tag = args.weights if args.weights != '' else None

        fast_test = args.fast_test
        n_skip_frames = args.n_skip_frames

        #log dir
        log_dir = os.path.join(config.cfg.LOG_DIR,'tracking', tag)


        tracklet_pred_dir = os.path.join(log_dir, 'tracklet')
        os.makedirs(tracklet_pred_dir, exist_ok=True)


        frame_offset = 0
        dataset_loader = None
        gt_tracklet_file = None

        # Set true if you want score after export predicted tracklet xml
        # set false if you just want to export tracklet xml
        if_score =False

        if config.cfg.DATA_SETS_TYPE == 'didi2':
            assert cfg.OBJ_TYPE == 'car' or cfg.OBJ_TYPE == 'ped'
            if cfg.OBJ_TYPE == 'car':
                test_bags = [
                    # 'test_car/ford01',
                    'test_car/ford02',
                    'test_car/ford03',
                    'test_car/ford04',
                    'test_car/ford05',
                    'test_car/ford06',
                    'test_car/ford07',
                    'test_car/mustang01'
                ]
            else:
                test_bags = [
                    'test_ped/ped_test',
                ]

        elif config.cfg.DATA_SETS_TYPE == 'didi':
            pass #todo
            # if_score = True
            # if 1:
            #     dataset = {'Round1Test': ['19_f2']}
            #
            # else:
            #     car = '3'
            #     data = '7'
            #     dataset = {
            #         car: [data]
            #     }
            #
            #     # compare newly generated tracklet_label_pred.xml with tracklet_labels_gt.xml. Change the path accordingly to
            #     #  fits you needs.
            #     gt_tracklet_file = os.path.join(cfg.RAW_DATA_SETS_DIR, car, data, 'tracklet_labels.xml')

        elif config.cfg.DATA_SETS_TYPE == 'kitti':
            if_score = False
            car = '2011_09_26'
            data = '0001'
            dataset = {
                car: [data]
            }



            # compare newly generated tracklet_label_pred.xml with tracklet_labels_gt.xml. Change the path accordingly to
            #  fits you needs.
            gt_tracklet_file = os.path.join(cfg.RAW_DATA_SETS_DIR, car, car + '_drive_' + data + '_sync',
                                            'tracklet_labels.xml')

        dataset_loader = ub.batch_loading(cfg.PREPROCESSED_DATA_SETS_DIR, dataset, is_testset=True)

        #print("tracklet_pred_dir: " + tracklet_pred_dir)
        pred_file = pred_and_save(tracklet_pred_dir, dataset_loader,
                                  frame_offset=0, log_tag=tag, weights_tag=weights_tag)
        #if if_score:
        tracklet_score(pred_file=pred_file, gt_file=gt_tracklet_file, output_dir=tracklet_pred_dir)
        print("scores are save under {} directory.".format(tracklet_pred_dir))

        print("Completed")

    elif args.data:
        print ('Ok! you runned the data')

        print('%s: calling main function ... ' % os.path.basename(__file__))
        if (cfg.DATA_SETS_TYPE == 'didi'):
            data_dir = {'1': ['15', '10']}
            data_dir = OrderedDict(data_dir)
            frames_index = None  # None
        elif (cfg.DATA_SETS_TYPE == 'didi2'):
            dir_prefix = '/home/stu/round12_data/raw/didi'

            bag_groups = ['suburu_pulling_to_left',
                          'nissan_following_long',
                          'suburu_following_long',
                          'nissan_pulling_to_right',
                          'suburu_not_visible',
                          'cmax_following_long',
                          'nissan_driving_past_it',
                          'cmax_sitting_still',
                          'suburu_pulling_up_to_it',
                          'suburu_driving_towards_it',
                          'suburu_sitting_still',
                          'suburu_driving_away',
                          'suburu_follows_capture',
                          'bmw_sitting_still',
                          'suburu_leading_front_left',
                          'nissan_sitting_still',
                          'nissan_brief',
                          'suburu_leading_at_distance',
                          'bmw_following_long',
                          'suburu_driving_past_it',
                          'nissan_pulling_up_to_it',
                          'suburu_driving_parallel',
                          'nissan_pulling_to_left',
                          'nissan_pulling_away', 'ped_train']

            bag_groups = ['suburu_pulling_to_left',
                          'nissan_following_long',
                          'nissan_driving_past_it',
                          'cmax_sitting_still',
                          'cmax_following_long',
                          'suburu_driving_towards_it',
                          'suburu_sitting_still',
                          'suburu_driving_away',
                          'suburu_follows_capture',
                          'bmw_sitting_still',
                          'suburu_leading_front_left',
                          'nissan_sitting_still',
                          'suburu_leading_at_distance',
                          'suburu_driving_past_it',
                          'nissan_pulling_to_left',
                          'nissan_pulling_away', 'ped_train']

            # use orderedDict to fix the dictionary order.
            data_dir = OrderedDict([(bag_group, None) for bag_group in bag_groups])
            print('ordered dictionary here: ', data_dir)

            frames_index = None  # None
        elif cfg.DATA_SETS_TYPE == 'kitti':
            data_dir = {'2011_09_26': ['0051']}

            frames_index = None  # [0,5,8,12,16,20,50]
        elif cfg.DATA_SETS_TYPE == 'test':
            data_dir = {'1': None, '2': None}
            data_dir = OrderedDict(data_dir)
            frames_index = None
        else:
            raise ValueError('unexpected type in cfg.DATA_SETS_TYPE item: {}!'.format(cfg.DATA_SETS_TYPE))


        t0 = time.time()

        preproces(data_dir, frames_index)

        print('use time : {}'.format(time.time() - t0))




