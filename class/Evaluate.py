import math
import numpy as np
from ipdb import set_trace


class Evaluate():
    """
    评估类，对算法的好坏进行评估
    """
    def __init__(self, conf):
        self.conf = conf

    def getIdcg(self, length):
        """
        功能函数：获取IDCG
        :param length:
        :return:
        """
        idcg = 0.0
        for i in range(length):
            idcg = idcg + math.log(2) / math.log(i + 2)
        return idcg

    def getDcg(self, value):
        """
        功能函数：获取DCG
        :param value:
        :return:
        """
        dcg = math.log(2) / math.log(value + 2)
        return dcg

    def getHr(self, value):
        hit = 1.0
        return hit

    def evaluateRankingPerformance(self, evaluate_index_dict, evaluate_real_rating_matrix, \
                                   evaluate_predict_rating_matrix, topK, num_procs, exp_flag=0, sp_name=None,
                                   result_file=None):
        """
        核心函数，评估算法好坏，返回 HR 和 NDCG
        :param evaluate_index_dict:
        :param evaluate_real_rating_matrix:
        :param evaluate_predict_rating_matrix:
        :param topK:
        :param num_procs:
        :param exp_flag:
        :param sp_name:
        :param result_file:
        :return:
        """
        user_list = list(evaluate_index_dict.keys())
        batch_size = int(len(user_list) / num_procs)

        hr_list, ndcg_list = [], []
        index = 0
        for _ in range(num_procs):  # 484
            if index + batch_size < len(user_list):
                batch_user_list = user_list[index:index + batch_size]
                index = index + batch_size
            else:
                batch_user_list = user_list[index:len(user_list)]
            tmp_hr_list, tmp_ndcg_list = self.getHrNdcgProc(evaluate_index_dict, evaluate_real_rating_matrix, \
                                                            evaluate_predict_rating_matrix, topK, batch_user_list)
            hr_list.extend(tmp_hr_list)
            ndcg_list.extend(tmp_ndcg_list)
        return np.mean(hr_list), np.mean(ndcg_list)

    def getHrNdcgProc(self,
                      evaluate_index_dict,
                      evaluate_real_rating_matrix,
                      evaluate_predict_rating_matrix,
                      topK,
                      user_list):
        """
        具体实现步骤
        :param evaluate_index_dict:
        :param evaluate_real_rating_matrix:
        :param evaluate_predict_rating_matrix:
        :param topK:
        :param user_list:
        :return:
        """

        tmp_hr_list, tmp_ndcg_list = [], []

        for u in user_list:
            real_item_index_list = evaluate_index_dict[u]
            real_item_rating_list = list(np.concatenate(evaluate_real_rating_matrix[real_item_index_list]))
            positive_length = len(real_item_rating_list)
            target_length = min(positive_length, topK)

            predict_rating_list = evaluate_predict_rating_matrix[u]
            real_item_rating_list.extend(predict_rating_list)
            sort_index = np.argsort(real_item_rating_list)
            sort_index = sort_index[::-1]

            user_hr_list = []
            user_ndcg_list = []
            hits_num = 0
            for idx in range(topK):
                ranking = sort_index[idx]
                if ranking < positive_length:
                    hits_num += 1
                    user_hr_list.append(self.getHr(idx))
                    user_ndcg_list.append(self.getDcg(idx))

            idcg = self.getIdcg(target_length)

            tmp_hr = np.sum(user_hr_list) / target_length
            tmp_ndcg = np.sum(user_ndcg_list) / idcg
            tmp_hr_list.append(tmp_hr)
            tmp_ndcg_list.append(tmp_ndcg)

        return tmp_hr_list, tmp_ndcg_list
