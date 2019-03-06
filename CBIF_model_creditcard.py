# encoding=utf-8
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab as pb
import sklearn
import seaborn
import scipy.io as scio
import datetime
from sklearn.cluster import KMeans
from sklearn import metrics
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest



def load_dataset():
    ccdata = pd.read_csv('duibi_dataset/creditcard.csv', low_memory=False)

    # print(ccdata.columns)
    # print(ccdata.shape)
    # print(ccdata.describe)

    count_fraud_trans = ccdata['Class'][ccdata['Class'] == 1].count()
    count_valid_trans = ccdata['Class'][ccdata['Class'] == 0].count()
    percent_outlier = count_fraud_trans / (count_valid_trans)

    return ccdata, percent_outlier

def set_k(ccdata):
    columns = ccdata.columns.tolist()
    columns_V_all = [c for c in columns if c not in ["Amount", "V22", "V23", "V24", "V25", "V26", "V27", "V28"]]
    columns_V_part = [c for c in columns_V_all if
                      c not in ["Class", "Amount", "V22", "V23", "V24", "V25", "V26", "V27", "V28"]]

    # # '利用SSE选择k'
    # SSE = []  # 存放每次结果的误差平方和
    # for k in range(1, 9):
    #     estimator = KMeans(n_clusters=k)  # 构造聚类器
    #     estimator.fit(ccdata[columns_V_part])
    #     SSE.append(estimator.inertia_)
    # X = range(1, 9)
    # plt.xlabel('k')
    # plt.ylabel('SSE')
    # plt.plot(X, SSE, 'o-')
    # plt.show()
    # plt.savefig('k-means.png')

    return columns_V_all, columns_V_part

def k_means_model(columns_V_all, columns_V_part, ccdata):
    ccdata = ccdata[columns_V_all]
    # ccdata = ccdata.iloc[:10000, :]

    k = 2  # 聚类的类别
    iteration = 30000  # 聚类最大循环次数
    model = KMeans(init='k-means++',n_clusters=k, n_jobs=50, max_iter=iteration)  # 分为k类,
    model.fit(ccdata[columns_V_part])  # 开始聚类

    # 简单打印结果
    r1 = pd.Series(model.labels_).value_counts()  # 统计各个类别的数目
    r2 = pd.DataFrame(model.cluster_centers_)  # 找出聚类中心
    df = pd.concat([r2, r1], axis=1)  # 横向连接(0是纵向), 得到聚类中心对应的类别下的数目

    # 详细输出原始数据及其类别
    df = pd.concat([ccdata, pd.Series(model.labels_, index=ccdata.index)], axis=1)  # 详细
    # 输出每个样本对应的类别
    df.columns = list(ccdata.columns) + [u'cluster_number']  # 重命名表头
    print('轮廓系数：%0.3f' % metrics.silhouette_score(df, df["cluster_number"], metric='sqeuclidean'))



    # 对不同的类进行分离
    cluster0_old = df[df['cluster_number'] == 0].copy()
    cluster1_old = df[df['cluster_number'] == 1].copy()
    cluster0 = cluster0_old
    cluster1 = cluster1_old

    keys = ["cluster0", "cluster1"]

    models = {"IF_0": IsolationForest(max_samples=len(cluster0), contamination=0.1, random_state=1, behaviour="new", n_estimators=20),
              "IF_1": IsolationForest(max_samples=len(cluster1), contamination=0.1, random_state=1, behaviour="new", n_estimators=20)}

    if keys[0] == "cluster0":
        model = models.get("IF_0")
        # IF计算运行时间
        start0 = datetime.datetime.now()
        model.fit(cluster0[columns_V_part])
        scores_pred_0 = model.decision_function(cluster0[columns_V_part])
        Y_pred_0 = model.predict(cluster0[columns_V_part])
        end0 = datetime.datetime.now()
        cluster0 = pd.concat([cluster0, pd.Series(scores_pred_0, index=cluster0.index)], axis=1)
        cluster0 = pd.concat([cluster0, pd.Series(Y_pred_0, index=cluster0.index)], axis=1)
        cluster0.columns = list(df.columns) + [u'scores_pred', u'Y_pred']


    if keys[1] == "cluster1":
        model = models.get("IF_1")
        # IF计算运行时间
        start1 = datetime.datetime.now()
        model.fit(cluster1[columns_V_part])
        scores_pred_1 = model.decision_function(cluster1[columns_V_part])
        Y_pred_1 = model.predict(cluster1[columns_V_part])
        end1 = datetime.datetime.now()
        cluster1 = pd.concat([cluster1, pd.Series(scores_pred_1, index=cluster1.index)], axis=1)
        cluster1 = pd.concat([cluster1, pd.Series(Y_pred_1, index=cluster1.index)], axis=1)
        cluster1.columns = list(df.columns) + [u'scores_pred', u'Y_pred']


    print("KMIF运行时间：", end0 - start0 + end1 - start1)

    # print("异常程度评分：", scores_pred_1)
    df = pd.concat([cluster0, cluster1])
    df.sort_index(inplace=True)
    # print('calinski_harabaz_score：%0.3f' % metrics.calinski_harabaz_score(df[columns_V_part], df["Y_pred"]))
    # print('davies_bouldin_score：%0.3f' % metrics.davies_bouldin_score(df[columns_V_part], df["Y_pred"]))
    # 默认情况下，这些模型的预测值给出-1和+1，需要将其更改为0和1

    df.loc[:, "Y_pred"][df["Y_pred"] == 1] = 0
    df.loc[:, "Y_pred"][df["Y_pred"] == -1] = 1

    print(df)


    error_count = (df["Y_pred"] != df["Class"]).sum()
    # Printing the metrics for the classification algorithms
    print("error_count: ", error_count)
    print("accuracy score: ", accuracy_score(df["Class"], df["Y_pred"]))

    print(classification_report(df["Class"], df["Y_pred"]))







    # return r
if __name__ == "__main__":
    # 1.加载数据集
    ccdata, percent_outlier = load_dataset()
    print("数据集加载完成。。。")

    # 2.手肘法选择k
    columns_V_all, columns_V_part = set_k(ccdata)

    # # 3.K-Means进行建模
    k_means_model(columns_V_all, columns_V_part, ccdata)
