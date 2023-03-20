import argparse
import os
import sys

def progressBar(i, max, text):
    """
    Print a progress bar during training.
    :param i: index of current iteration/epoch.
    :param max: max number of iterations/epochs.
    :param text: Text to print on the right of the progress bar.
    :return: None
    """
    bar_size = 60
    j = (i+1) / max
    sys.stdout.write('\r')
    sys.stdout.write(
        f"[{'=' * int(bar_size * j):{bar_size}s}] {int(100 * j)}%  {text}")
    sys.stdout.flush()

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
        progressBar(epoch - FLAGS.start_epoch, FLAGS.end_epoch + 1 - FLAGS.start_epoch, 'remapping ' + str(epoch))
        label_folder = FLAGS.experiment_label_folder + '/' + 'model_epoch_' + str(epoch) + '/point_predict'
        command = 'python remap_semantic_labels.py --predictions ../semantic_kitti/predictions --split valid --inverse --label_folder ' + label_folder
        os.system(command)

