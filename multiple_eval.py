import argparse
import os
import tensorboard
import subprocess
import shlex

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

    parser.add_argument("--subprocess", action="store_true")

    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.end_epoch == -1:
        FLAGS.end_epoch = FLAGS.start_epoch

    output_file_list = []
    command_list = []

    for epoch in range(FLAGS.start_epoch, FLAGS.end_epoch + 1):
        if FLAGS.subprocess:
            dir = 'outputs'
            sub_dir = FLAGS.experiment_label_folder
            file_name = 'epoch_' + str(epoch) + '.txt'
            command = 'python evaluate_semantics.py --dataset ../semantic_kitti/dataset --predictions ../semantic_kitti/predictions --split valid --prediction_source_folder ' + FLAGS.experiment_label_folder + '/' + 'model_epoch_' + str(epoch)
            if not os.path.exists(dir + '/' + sub_dir):
                os.mkdir(dir + '/' + sub_dir)
            args = shlex.split(command)
            args[0] = '/data1/liyang/anaconda3/envs/yang_real/bin/python'

            # output file
            f = open(dir + '/' + sub_dir + '/' + file_name, "w")

            # start process
            # subprocess.call(args, stdout=f)
            # abcd = 1

            output_file_list.append(f)
            command_list.append(args)


        else:
            command = 'python evaluate_semantics.py --dataset ../semantic_kitti/dataset --predictions ../semantic_kitti/predictions --split valid --prediction_source_folder ' + FLAGS.experiment_label_folder + '/' + 'model_epoch_' + str(epoch)
            os.system(command)


    if FLAGS.subprocess:

        procs = []
        for i in range(len(command_list)):
            command = command_list[i]
            f = output_file_list[i]
            procs.append(subprocess.Popen(command, stdout=f))

        for p in procs:
            p.wait()

        abcd = 1