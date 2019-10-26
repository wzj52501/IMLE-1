import tensorflow as tf
from model import GLANN
import sys


parser = argparse.ArgumentParser()

parser.add_argument("-f", "--flag", type=distutils.util.strtobool, default='true')
parser.add_argument("-g", "--gpu_number", type=str, default="0")
parser.add_argument("-p", "--project", type=str, default="began")

# Train Data
parser.add_argument("-d", "--data_dir", type=str, default="./Data")
parser.add_argument("-trd", "--dataset", type=str, default="celeba")
parser.add_argument("-tro", "--data_opt", type=str, default="crop")
parser.add_argument("-trs", "--data_size", type=int, default=64)

# Train Iteration
parser.add_argument("-n", "--niter", type=int, default=50)
parser.add_argument("-ns", "--nsnapshot", type=int, default=2440)
parser.add_argument("-mx", "--max_to_keep", type=int, default=5)

# Train Parameter
parser.add_argument("-b", "--batch_size", type=int, default=16)
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
parser.add_argument("-m", "--momentum", type=float, default=0.5)
parser.add_argument("-m2", "--momentum2", type=float, default=0.999)
parser.add_argument("-gm", "--gamma", type=float, default=0.5)
parser.add_argument("-lm", "--lamda", type=float, default=0.001)
parser.add_argument("-fn", "--filter_number", type=int, default=64)
parser.add_argument("-z", "--input_size", type=int, default=64)
parser.add_argument("-em", "--embedding", type=int, default=64)

args = parser.parse_args()

gpu_number = args.gpu_number

if __name__ == '__main__:
    if(sys.platform == 'darwin'):
        device = 'cpu'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
        device = 'gpu'

    with tf.Session() as sess:

        model = GLANN(args,sess)
        model.train()
