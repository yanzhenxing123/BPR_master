# BPR_master
Recommended system algorithm implementation

## BPR (Bayesian Personalized Ranking)
[BPR原理链接](https://zhuanlan.zhihu.com/p/60704781)

贝叶斯个性化排序：通过贝叶斯公式的应用，去寻找一个最优的预测排名函数，其中包括用户和物品的特征向量。
可以解决难以处理的数据稀疏性和时效性问题，并且可以根据个人偏好为不同的用户提供不同的推荐策略.


## 推荐系统评估指标
[HR和NDCG说明链接](https://blog.csdn.net/shiaiao/article/details/109004341)

评估指标使用HR和NDCG

在多次模型评估后，可以发现HR和NDCG有明显的提升。

## 代码执行
版本：`python=3.6 tensorflow=1.12.0`

```shell script
pip install -r conf/requirements.txt
python entry.py --data_name=yelp --model_name=BPR
```