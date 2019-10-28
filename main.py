import tensorflow as tf
from glann import GLANN
import argparse
import sys


parser = argparse.ArgumentParser()

# parser.add_argument("-b", "--batch_normal", type=distutils.util.strtobool, default='true')
parser.add_argument("-g", "--gpu_number", type=str, default="0")
parser.add_argument("-p", "--project", type=str, default="began")

# Train Data
parser.add_argument("-d", "--data_dir", type=str, default="./Data")
parser.add_argument("-trd", "--dataset", type=str, default="celeba")
parser.add_argument("-tro", "--data_opt", type=str, default="crop")
parser.add_argument("-trs", "--data_size", type=int, default=64)

# Train Iteration
parser.add_argument("-b", "--batch_size", type=int, default=16)
parser.add_argument("-ndeep", "--n_deep", type=int, default=500)
parser.add_argument("-ndim", "--n_dim", type=int, default=100)
parser.add_argument("-inh", "--image_height", type=int, default=250)
parser.add_argument("-inw", "--image_weight", type=int, default=250)
parser.add_argument("-ind", "--input_deep", type=int, default=100)
parser.add_argument("-e", "--epoch", type=int, default=1)
parser.add_argument("-gn", "--gpu_nums", type=int, default=3)
parser.add_argument("-tu", "--train_utils", type=str, default='cpu')


# Train Parameter
parser.add_argument("-l", "--lr", type=float, default=1e-4)
# parser.add_argument("-m", "--momentum", type=float, default=0.5)
# parser.add_argument("-m2", "--momentum2", type=float, default=0.999)
# parser.add_argument("-gm", "--gamma", type=float, default=0.5)
# parser.add_argument("-lm", "--lamda", type=float, default=0.001)
# parser.add_argument("-fn", "--filter_number", type=int, default=64)
# parser.add_argument("-z", "--input_size", type=int, default=64)
# parser.add_argument("-em", "--embedding", type=int, default=64)

args = parser.parse_args()

gpu_number = args.gpu_number

if __name__ == '__main__':
    if(sys.platform == 'darwin'):
        device = 'cpu'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
        device = 'gpu'

    with tf.Session() as sess:
        model = GLANN(args,sess)
        model.main()
