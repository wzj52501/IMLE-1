import tensorflow as tf
from glann import GLANN
import argparse
import sys


parser = argparse.ArgumentParser()

# Train Data
parser.add_argument("-dd", "--data_dir", type=str, default="./Data")
parser.add_argument("-sd", "--summary_dir", type=str, default="./logs")
parser.add_argument("-ms", "--model_save_path", type=str, default="./model")

# Train Iteration
parser.add_argument("-b", "--batch_size", type=int, default=16)
parser.add_argument("-ndeep", "--n_deep", type=int, default=64)
parser.add_argument("-ndim", "--n_dim", type=int, default=64)
parser.add_argument("-inh", "--image_height", type=int, default=256)
parser.add_argument("-inw", "--image_weight", type=int, default=256)
parser.add_argument("-ind", "--input_deep", type=int, default=64)
parser.add_argument("-e", "--epoch", type=int, default=1)
parser.add_argument("-gn", "--gpu_nums", type=int, default=2)
parser.add_argument("-tu", "--train_utils", type=str, default='cpu')

# Train Parameter
parser.add_argument("-l", "--lr", type=float, default=1e-4)



args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num

if __name__ == '__main__':
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config = config) as sess:
        model = GLANN(args,sess)
        model.main()
