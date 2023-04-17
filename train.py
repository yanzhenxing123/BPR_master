from __future__ import division
import os, sys, shutil

from time import time
import numpy as np
import tensorflow as tf
from ipdb import set_trace

sys.path.append(os.path.join(os.getcwd(), 'class'))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore the warnings

from Logging import Logging


def prepareModelSupplement(conf, data, model, evaluate):
    """
    准备模型补充
    :param conf: 配置文件
    :param data: 数据集 yelp or flickr
    :param model: 模型 BPR or PMF
    :param evaluate: 评估指标 BPR or NDCG
    :return: None
    """

    # 1. 日志准备
    scene = ""
    log_dir = os.path.join(os.getcwd(), 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # define log name 
    log_path = os.path.join(os.getcwd(), 'log/%s_%s.log' % (conf.data_name, conf.model_name))

    # 2. start to prepare data for training and evaluating
    # 调用了   self.createTrainHandle()  创建训练数据
    #         self.createEvaluateHandle()  创建评估数据
    data.initializeRankingHandle()  # 创建训练数据和评估数据

    # S1
    print('System start to load data...')
    t0 = time()
    # train validation test test_evaluate 都是DataModule对象
    d_train, d_val, d_test, d_test_eva = data.train, data.val, data.test, data.test_eva

    # 数据集的初始化
    d_train.initializeRankingTrain()  # 初始化训练集
    d_val.initializeRankingVT()  # 初始化验证集
    d_test.initializeRankingVT()  # 初始化测试集
    d_test_eva.initalizeRankingEva()  # 初始化评估数据

    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1 - t0))

    # 3. prepare model necessary data. 准备模型需要的数据
    data_dict = d_train.prepareModelSupplement(model)

    model.inputSupply(data_dict)  # 取出数据（为模型提供上面准备好的数据）
    model.startConstructGraph()  # 创建图（定义节点等等）

    # standard tensorflow running environment initialize tensorflow运行环境初始化
    tf_conf = tf.ConfigProto()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_conf)
    sess.run(model.init)

    if conf.pretrain_flag == 1:
        model.saver.restore(sess, conf.pre_model)

    # set debug_flag=0, doesn't print any results
    log = Logging(log_path)

    log.record('Following will output the evaluation of the model:')

    # 4. Start Training !!! 开始训练
    for epoch in range(1, conf.epochs + 1):
        # optimize model with training data and compute train loss 利用训练数据优化模型并计算训练损失
        tmp_train_loss = []
        t0 = time()

        # tmp_total_list = []
        while d_train.terminal_flag:
            d_train.getTrainRankingBatch()
            d_train.linkedMap()
            # set_trace()

            train_feed_dict = {}
            for (key, value) in model.map_dict['train'].items():
                train_feed_dict[key] = d_train.data_dict[value]

            [sub_train_loss, _] = sess.run([model.map_dict['out']['train'], model.opt], feed_dict=train_feed_dict)
            tmp_train_loss.append(sub_train_loss)

        train_loss = np.mean(tmp_train_loss)
        t1 = time()

        # compute val loss and test loss
        d_val.getVTRankingOneBatch()
        d_val.linkedMap()
        val_feed_dict = {}
        for (key, value) in model.map_dict['val'].items():
            val_feed_dict[key] = d_val.data_dict[value]
        val_loss = sess.run(model.map_dict['out']['val'], feed_dict=val_feed_dict)

        d_test.getVTRankingOneBatch()
        d_test.linkedMap()
        test_feed_dict = {}
        for (key, value) in model.map_dict['test'].items():
            test_feed_dict[key] = d_test.data_dict[value]
        test_loss = sess.run(model.map_dict['out']['test'], feed_dict=test_feed_dict)
        t2 = time()

        tt2 = time()

        # start evaluate model performance, hr and ndcg 评估指标
        def getPositivePredictions():
            """
            获得正面预测
            """
            d_test_eva.getEvaPositiveBatch()
            d_test_eva.linkedRankingEvaMap()
            eva_feed_dict = {}
            for (key, value) in model.map_dict['eva'].items():
                eva_feed_dict[key] = d_test_eva.data_dict[value]
            positive_predictions = sess.run(
                model.map_dict['out']['eva'],
                feed_dict=eva_feed_dict
            )
            return positive_predictions

        def getNegativePredictions():
            """
            获取负面预测
            """
            negative_predictions = {}
            terminal_flag = 1
            while terminal_flag:
                batch_user_list, terminal_flag = d_test_eva.getEvaRankingBatch()
                d_test_eva.linkedRankingEvaMap()

                eva_feed_dict = {}
                for (key, value) in model.map_dict['eva'].items():
                    eva_feed_dict[key] = d_test_eva.data_dict[value]
                index = 0
                tmp_negative_predictions = np.reshape(
                    sess.run(
                        model.map_dict['out']['eva'],
                        feed_dict=eva_feed_dict
                    ),
                    [-1, conf.evaluate])
                for u in batch_user_list:
                    negative_predictions[u] = tmp_negative_predictions[index]
                    index = index + 1
            return negative_predictions

        index_dict = d_test_eva.eva_index_dict

        positive_predictionnum_s = getPositivePredictions()
        negative_predictions = getNegativePredictions()

        d_test_eva.index = 0  # !!!important, prepare for new batch

        tt3 = time()
        # s1
        hr, ndcg = evaluate.evaluateRankingPerformance(index_dict, positive_predictions, negative_predictions,
                                                       conf.topk, conf.num_procs)
        hr_5, ndcg_5 = evaluate.evaluateRankingPerformance(index_dict, positive_predictions, negative_predictions,
                                                           conf.top5, conf.num_procs)
        hr_15, ndcg_15 = evaluate.evaluateRankingPerformance(index_dict, positive_predictions, negative_predictions,
                                                             conf.top15, conf.num_procs)

        log.record('Epoch:%d, compute loss cost:%.4fs, train loss:%.4f, val loss:%.4f, test loss:%.4f' % (
            epoch, (t2 - t0), train_loss, val_loss, test_loss))
        log.record(
            'Evaluate cost:%.4fs \n Top5: hr:%.4f, ndcg:%.4f \n Top10: hr:%.4f, ndcg:%.4f \n Top15: hr:%.4f, ndcg:%.4f' % (
                (tt3 - tt2), hr_5, ndcg_5, hr, ndcg, hr_15, ndcg_15)
        )

        d_train.generateTrainNegative()
