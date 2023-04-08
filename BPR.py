from __future__ import division
import tensorflow as tf
import numpy as np
from ipdb import set_trace


class BPR():
    """
    BPR模型实现
    """

    def __init__(self, conf):
        self.conf = conf
        self.supply_set = (
            'SOCIAL_NEIGHBORS_SPARSE_MATRIX',  # 社交邻居稀疏矩阵
            'CONSUMED_ITEMS_SPARSE_MATRIX',  # user_item 稀疏矩阵
            'ITEM_CUSTOMER_SPARSE_MATRIX',  # item_user 稀疏矩阵
            'FEATURE_SPARSE_MATRIX',  # 特征_解析_矩阵
            'COLD_START_MATRIX'  # 冷启动矩阵
        )

    def startConstructGraph(self):
        self.initializeNodes()
        self.constructTrainGraph()
        self.saveVariables()
        self.defineMap()

        # self.user_social_neighbor_low_att_list = tf.Variable(tf.random_normal([len(self.social_neighbors_indices_list)], stddev=0.01))
        # self.user_consumption_items_low_att_list = tf.Variable(tf.random_normal([len(self.social_neighbors_indices_list)], stddev=0.01))
        # self.item_users_low_att_list = tf.Variable(tf.random_normal([len(self.item_customer_indices_list)], stddev=0.01))

    def inputSupply(self, data_dict):
        # user-item

        self.consumed_items_indices_input = data_dict['CONSUMED_ITEMS_INDICES_INPUT']
        self.consumed_items_values_input = data_dict['CONSUMED_ITEMS_VALUES_INPUT']

        # item-user
        self.item_customer_indices_input = data_dict['ITEM_CUSTOMER_INDICES_INPUT']
        self.item_customer_values_input = data_dict['ITEM_CUSTOMER_VALUES_INPUT']

        # prepare sparse matrix, in order to compute user's embedding from social neighbors and consumed items
        self.social_neighbors_dense_shape = np.array([self.conf.num_users, self.conf.num_users]).astype(np.int64)
        self.consumed_items_dense_shape = np.array([self.conf.num_users, self.conf.num_items]).astype(np.int64)
        self.item_customer_dense_shape = np.array([self.conf.num_items, self.conf.num_users]).astype(np.int64)

    def initializeNodes(self):
        self.item_input = tf.placeholder("int32", [None, 1])  # Get item embedding from the core_item_input
        self.user_input = tf.placeholder("int32", [None, 1])  # Get user embedding from the core_user_input
        self.labels_input = tf.placeholder("float32", [None, 1])

        self.user_embedding = tf.Variable(
            tf.random_normal([self.conf.num_users, self.conf.dimension], stddev=0.01), name='user_embedding')
        self.item_embedding = tf.Variable(
            tf.random_normal([self.conf.num_items, self.conf.dimension], stddev=0.01), name='item_embedding')

        self.item_fusion_layer = tf.layers.Dense( \
            self.conf.dimension, activation=tf.nn.sigmoid, name='item_fusion_layer')
        self.user_fusion_layer = tf.layers.Dense( \
            self.conf.dimension, activation=tf.nn.sigmoid, name='user_fusion_layer')

        self.reduce_dimension_layer = tf.layers.Dense( \
            self.conf.dimension, activation=tf.nn.sigmoid, name='reduce_dimension_layer')

    def constructTrainGraph(self):
        # handle review information, map the origin review into the new space and

        self.fusion_item_embedding = self.item_embedding
        self.fusion_user_embedding = self.user_embedding

        # BPR
        latest_user_latent = tf.gather_nd(self.user_embedding, self.user_input)
        latest_item_latent = tf.gather_nd(self.item_embedding, self.item_input)

        # mul_latent = tf.multiply(latest_user_latent, latest_item_latent)

        self.predict_vector = tf.multiply(latest_user_latent, latest_item_latent)
        self.prediction = tf.sigmoid(tf.reduce_sum(self.predict_vector, 1, keepdims=True))

        # self.prediction = self.predict_rating_layer(tf.concat([latest_user_latent, latest_item_latent], 1))

        self.loss = tf.nn.l2_loss(self.labels_input - self.prediction)

        self.opt_loss = tf.nn.l2_loss(self.labels_input - self.prediction)
        self.opt = tf.train.AdamOptimizer(self.conf.learning_rate).minimize(self.opt_loss)
        self.init = tf.global_variables_initializer()

    def saveVariables(self):
        ############################# Save Variables #################################
        variables_dict = {}
        variables_dict[self.user_embedding.op.name] = self.user_embedding
        variables_dict[self.item_embedding.op.name] = self.item_embedding

        for v in self.reduce_dimension_layer.variables:
            variables_dict[v.op.name] = v

        self.saver = tf.train.Saver(variables_dict)
        ############################# Save Variables #################################

    def defineMap(self):
        map_dict = {}
        map_dict['train'] = {
            self.user_input: 'USER_LIST',
            self.item_input: 'ITEM_LIST',
            self.labels_input: 'LABEL_LIST'
        }

        map_dict['val'] = {
            self.user_input: 'USER_LIST',
            self.item_input: 'ITEM_LIST',
            self.labels_input: 'LABEL_LIST'
        }

        map_dict['test'] = {
            self.user_input: 'USER_LIST',
            self.item_input: 'ITEM_LIST',
            self.labels_input: 'LABEL_LIST'
        }

        map_dict['eva'] = {
            self.user_input: 'EVA_USER_LIST',
            self.item_input: 'EVA_ITEM_LIST'
        }

        map_dict['out'] = {
            'train': self.loss,
            'val': self.loss,
            'test': self.loss,
            'eva': self.prediction,
        }

        self.map_dict = map_dict
