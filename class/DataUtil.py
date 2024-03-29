import os
from time import time
from DataModule import DataModule


class DataUtil():
    """
    处理数据工具类，DataUtil封装DataModule
    """

    def __init__(self, conf):
        self.conf = conf
        # print('DataUtil, Line12, test- conf data_dir:%s' % self.conf.data_dir)

    def initializeRankingHandle(self):
        """
        start 初始化排名句柄
        :return:
        """
        # t0 = time()
        self.createTrainHandle()
        self.createEvaluateHandle()
        # t1 = time()
        # print('Prepare data cost:%.4fs' % (t1 - t0))

    def createTrainHandle(self):
        """
        创建训练数据
        :return:
        """
        data_dir = self.conf.data_dir
        train_filename = "%s/%s.train.rating" % (data_dir, self.conf.data_name)
        val_filename = "%s/%s.val.rating" % (data_dir, self.conf.data_name)
        test_filename = "%s/%s.test.rating" % (data_dir, self.conf.data_name)

        self.train = DataModule(self.conf, train_filename)  # 训练集，模型选择和调优，在训练阶段
        self.val = DataModule(self.conf, val_filename)  # 验证集
        self.test = DataModule(self.conf, test_filename)  # 测试集

    def createEvaluateHandle(self):
        """
        创建评估数据
        :return:
        """
        data_dir = self.conf.data_dir
        val_filename = "%s/%s.val.rating" % (data_dir, self.conf.data_name)
        test_filename = "%s/%s.test.rating" % (data_dir, self.conf.data_name)

        self.val_eva = DataModule(self.conf, val_filename)
        self.test_eva = DataModule(self.conf, test_filename)
