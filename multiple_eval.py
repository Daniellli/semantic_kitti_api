import argparse
import os
import tensorboard

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./multiple_remap.py")
    parser.add_argument(
        '--experiment_label_folder',
        type=str,
        required=True,
        default=None,
        help='.'
    )

    parser.add_argument(
        '--start_epoch',
        type=int,
        required=False,
        default=0,
        help='.'
    )

    parser.add_argument(
        '--end_epoch',
        type=int,
        required=False,
        default=-1,
        help='.'
    )

    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.end_epoch == -1:
        FLAGS.end_epoch = FLAGS.start_epoch

    for epoch in range(FLAGS.start_epoch, FLAGS.end_epoch + 1):
        command = 'python evaluate_semantics.py --dataset ../semantic_kitti/dataset --predictions ../semantic_kitti/predictions --split valid --prediction_source_folder ' + FLAGS.experiment_label_folder + '/' + 'model_epoch_' + str(epoch)
        os.system(command)