import sys, os, argparse

sys.path.append(os.path.join(os.getcwd(), 'class'))

from ParserConf import ParserConf
from DataUtil import DataUtil
from Evaluate import Evaluate
import pdb

from BPR import BPR

from ipdb import set_trace


def executeTrainModel(config_path, model_name):
    """
    执行训练模型
    :param config_path: 配置文件路径
    :param model_name: 模型名称 BPR or PMF
    :return:
    """
    print(config_path)
    # print('System start to prepare parser config file...')
    conf = ParserConf(config_path)
    conf.parserConf()

    # print conf.topk

    # print('System start to load TensorFlow graph...')

    model = eval(model_name)
    model = model(conf)  # eg: model = BPR(conf)

    # print('System start to load data...')
    data = DataUtil(conf)
    evaluate = Evaluate(conf)

    import train as starter
    starter.prepareModelSupplement(conf, data, model, evaluate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Welcome to the Experiment Platform Entry')
    parser.add_argument('--data_name', nargs='?', help='data name')
    parser.add_argument('--model_name', nargs='?', help='model name')
    parser.add_argument('--gpu', nargs='?', help='available gpu id')

    args = parser.parse_args()

    data_name = args.data_name
    # set_trace()
    # set_trace()
    model_name = args.model_name
    device_id = args.gpu

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    config_path = os.path.join(os.getcwd(), 'conf/%s_%s.ini' % (data_name, model_name))

    executeTrainModel(config_path, model_name)
