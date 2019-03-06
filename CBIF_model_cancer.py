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
    breastw_path = 'duibi_dataset/breastw.mat'
    data = scio.loadmat(breastw_path)
    data_train_data = data.get('X')  # 取出字典里的data
    data_train_label = data.get('y')  # 取出字典里的label

    ccdata = pd.DataFrame(data_train_data)
    ccdata.columns = ['C_D', 'C_T', 'U_C_Si', 'U_C_Sh', 'M_A','S_E_C_S', 'B_N', 'B_C', 'N_N']
    ccdata_label = pd.DataFrame(data_train_label)
    ccdata_label.columns = ['Class']
    ccdata_new = pd.concat([ccdata, ccdata_label], axis=1) #横向合并

    count_fraud_trans = ccdata_new['Class'][ccdata_new['Class'] == 1].count()
    count_valid_trans = ccdata_new['Class'][ccdata_new['Class'] == 0].count()
    percent_outlier = count_fraud_trans / (count_valid_trans)

    return ccdata, ccdata_new, percent_outlier

def set_k(ccdata):
    # '利用SSE选择k'
    SSE = []  # 存放每次结果的误差平方和
    for k in range(1, 9):
        estimator = KMeans(n_clusters=k)  # 构造聚类器
        estimator.fit(ccdata[['C_D', 'C_T', 'U_C_Si', 'U_C_Sh', 'M_A','S_E_C_S', 'B_N', 'B_C', 'N_N']])
        SSE.append(estimator.inertia_)
    X = range(1, 9)
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.plot(X, SSE, 'o-')
    plt.show()

def k_means_model(ccdata_new):

    # 从数据框中提取列。
    columns = ccdata_new.columns.tolist()
    columns_V = [c for c in columns if c not in ["Class"]]

    k = 2  # 聚类的类别
    iteration = 30000  # 聚类最大循环次数
    model = KMeans(init='k-means++',n_clusters=k, n_jobs=50, max_iter=iteration)  # 分为k类,
    model.fit(ccdata_new[columns_V])  # 开始聚类

    # 简单打印结果
    r1 = pd.Series(model.labels_).value_counts()  # 统计各个类别的数目
    r2 = pd.DataFrame(model.cluster_centers_)  # 找出聚类中心
    df = pd.concat([r2, r1], axis=1)  # 横向连接(0是纵向), 得到聚类中心对应的类别下的数目
    # print(df)

    # 详细输出原始数据及其类别
    df = pd.concat([ccdata_new, pd.Series(model.labels_, index=ccdata_new.index)], axis=1)  # 详细
    # 输出每个样本对应的类别
    df.columns = list(ccdata_new.columns) + [u'cluster_number']  # 重命名表头
    print('轮廓系数：%0.3f' % metrics.silhouette_score(df, df["cluster_number"], metric='sqeuclidean'))

    # 对不同的类进行分离
    cluster0_old = df[df['cluster_number'] == 0].copy()
    cluster1_old = df[df['cluster_number'] == 1].copy()
    cluster0 = cluster0_old
    cluster1 = cluster1_old

    keys = ["cluster0", "cluster1"]

    models = {"IF_0": IsolationForest(max_samples=len(cluster0), contamination=0.6, random_state=1, behaviour="new", n_estimators=20),
              "IF_1": IsolationForest(max_samples=len(cluster1), contamination=0.3, random_state=1, behaviour="new", n_estimators=20)}

    if keys[0] == "cluster0":
        model = models.get("IF_0")
        # IF计算运行时间
        start0 = datetime.datetime.now()
        # 这里columns_V还应除去cluster_number
        model.fit(cluster0[columns_V])
        scores_pred_0 = model.decision_function(cluster0[columns_V])
        Y_pred_0 = model.predict(cluster0[columns_V])
        end0 = datetime.datetime.now()
        cluster0 = pd.concat([cluster0, pd.Series(scores_pred_0, index=cluster0.index)], axis=1)
        cluster0 = pd.concat([cluster0, pd.Series(Y_pred_0, index=cluster0.index)], axis=1)
        cluster0.columns = list(df.columns) + [u'scores_pred', u'Y_pred']


    if keys[1] == "cluster1":
        model = models.get("IF_1")
        # IF计算运行时间
        start1 = datetime.datetime.now()
        model.fit(cluster1[columns_V])
        scores_pred_1 = model.decision_function(cluster1[columns_V])
        Y_pred_1 = model.predict(cluster1[columns_V])
        end1 = datetime.datetime.now()
        cluster1 = pd.concat([cluster1, pd.Series(scores_pred_1, index=cluster1.index)], axis=1)
        cluster1 = pd.concat([cluster1, pd.Series(Y_pred_1, index=cluster1.index)], axis=1)
        cluster1.columns = list(df.columns) + [u'scores_pred', u'Y_pred']


    print("IF运行时间：", end0 - start0 + end1 - start1)
    # print("异常程度评分：", scores_pred_1)
    df = pd.concat([cluster0, cluster1])
    df.sort_index(inplace=True)
    print('calinski_harabaz_score：%0.3f' % metrics.calinski_harabaz_score(df[columns_V], df["Y_pred"]))
    print('davies_bouldin_score：%0.3f' % metrics.davies_bouldin_score(df[columns_V], df["Y_pred"]))
    # 默认情况下，这些模型的预测值给出-1和+1，需要将其更改为0和1

    df.loc[:, "Y_pred"][df["Y_pred"] == 1] = 0
    df.loc[:, "Y_pred"][df["Y_pred"] == -1] = 1

    # print(df)
    error_count = (df["Y_pred"] != df["Class"]).sum()
    # Printing the metrics for the classification algorithms
    print("error_count: ", error_count)
    print("accuracy score: ", accuracy_score(df["Class"], df["Y_pred"]))
    print(classification_report(df["Class"], df["Y_pred"]))

    # return r
if __name__ == "__main__":
    # 1.加载数据集
    ccdata, ccdata_new, percent_outlier = load_dataset()
    print("数据集加载完成。。。")

    # 2.手肘法选择k
    # set_k(ccdata)

    # 3.K-Means进行建模
    k_means_model(ccdata_new)

