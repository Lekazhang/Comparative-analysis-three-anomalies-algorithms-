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
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn import datasets
from sklearn import preprocessing

def byte_decoder(val):
    # decodes byte literals to strings

    return val.decode('utf-8')

def load_dataset():
    target = 'target'
    sf = datasets.fetch_kddcup99(subset='SF', percent10=False)
    dfSF = pd.DataFrame(sf.data,
                        columns=["duration", "service", "src_bytes", "dst_bytes"])
    assert len(dfSF) > 0, "SF dataset no loaded."

    dfSF[target] = sf.target
    return target, dfSF

def data_process(target,dfSF):
    # 将热编码应用于string类型的字段
    # 将所有异常目标类型转换为单个异常类
    toDecodeSF = ['service', target]

    dfSF['binary_target'] = [0 if x == b'normal.' else 1 for x in dfSF[target]]

    leSF = preprocessing.LabelEncoder()

    for f in toDecodeSF:
        dfSF[f] = list(map(byte_decoder, dfSF[f]))
        dfSF[f] = leSF.fit_transform(dfSF[f])

    dfSF_normed = preprocessing.normalize(dfSF.drop([target, 'binary_target'], axis=1))

    return dfSF,dfSF_normed

def set_k(ccdata):

    # '利用SSE选择k'
    SSE = []  # 存放每次结果的误差平方和
    for k in range(1, 9):
        estimator = KMeans(n_clusters=k)  # 构造聚类器
        estimator.fit(ccdata)
        SSE.append(estimator.inertia_)
    X = range(1, 9)
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.plot(X, SSE, 'o-')
    plt.show()
    plt.savefig('k-means.png')



def k_means_model(ccdata, dfSF):
    ccdata = pd.DataFrame(ccdata,
                        columns=["duration", "service", "src_bytes", "dst_bytes"])

    # ccdata = ccdata.iloc[:10000, :]
    # dfSF = dfSF.iloc[:10000, :]

    # k = 3  # 聚类的类别
    # iteration = 30000  # 聚类最大循环次数
    # model = KMeans(init='k-means++',n_clusters=k, n_jobs=50, max_iter=iteration)  # 分为k类,
    # model.fit(ccdata)  # 开始聚类
    #
    # # 简单打印结果
    # r1 = pd.Series(model.labels_).value_counts()  # 统计各个类别的数目
    # r2 = pd.DataFrame(model.cluster_centers_)  # 找出聚类中心
    # df = pd.concat([r2, r1], axis=1)  # 横向连接(0是纵向), 得到聚类中心对应的类别下的数目
    #
    # # 详细输出原始数据及其类别
    # df = pd.concat([ccdata, pd.Series(model.labels_, index=ccdata.index)], axis=1)  # 详细
    # # 输出每个样本对应的类别
    # df.columns = list(ccdata.columns) + [u'cluster_number']  # 重命名表头
    # print('轮廓系数：%0.3f' % metrics.silhouette_score(df, df["cluster_number"], metric='sqeuclidean'))
    # # return 'ok'

    df = pd.read_csv('duibi_dataset/KDDCUP99_kmeans.csv', index_col=0)
    # 对不同的类进行分离
    cluster0_old = df[df['cluster_number'] == 0].copy()
    cluster1_old = df[df['cluster_number'] == 1].copy()
    cluster2_old = df[df['cluster_number'] == 2].copy()
    cluster0 = cluster0_old
    cluster1 = cluster1_old
    cluster2 = cluster2_old

    keys = ["cluster0", "cluster1", "cluster2"]

    models = {"IF_0": IsolationForest(max_samples=256, contamination=0.05, random_state=1, behaviour="new", n_estimators=20),
              "IF_1": IsolationForest(max_samples=256, contamination=0.05, random_state=1, behaviour="new", n_estimators=20),
              "IF_2": IsolationForest(max_samples=256, contamination=0.1, random_state=1, behaviour="new", n_estimators=20),}
    # 从数据框中提取列。
    columns = ccdata.columns.tolist()
    columns_V = [c for c in columns if c not in ["cluster_number"]]

    if keys[0] == "cluster0":
        model = models.get("IF_0")
        # IF计算运行时间
        start0 = datetime.datetime.now()
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

    if keys[2] == "cluster2":
        model = models.get("IF_2")
        # IF计算运行时间
        start2 = datetime.datetime.now()
        model.fit(cluster2[columns_V])
        scores_pred_2 = model.decision_function(cluster2[columns_V])
        Y_pred_2 = model.predict(cluster2[columns_V])
        end2 = datetime.datetime.now()
        cluster2 = pd.concat([cluster2, pd.Series(scores_pred_2, index=cluster2.index)], axis=1)
        cluster2 = pd.concat([cluster2, pd.Series(Y_pred_2, index=cluster2.index)], axis=1)
        cluster2.columns = list(df.columns) + [u'scores_pred', u'Y_pred']

    print("IF运行时间：", end0 - start0 + end1 - start1 + end2 - start2)

    # print("异常程度评分：", scores_pred_1)
    df = pd.concat([cluster0, cluster1, cluster2])
    df.sort_index(inplace=True)
    print('calinski_harabaz_score：%0.3f' % metrics.calinski_harabaz_score(df[columns_V], df["Y_pred"]))
    print('davies_bouldin_score：%0.3f' % metrics.davies_bouldin_score(df[columns_V], df["Y_pred"]))
    # 默认情况下，这些模型的预测值给出-1和+1，需要将其更改为0和1

    df.loc[:, "Y_pred"][df["Y_pred"] == 1] = 0
    df.loc[:, "Y_pred"][df["Y_pred"] == -1] = 1

    # print(df)

    error_count = (df["Y_pred"] != dfSF["binary_target"]).sum()
    # Printing the metrics for the classification algorithms
    print("error_count: ", error_count)
    print("accuracy score: ", accuracy_score(dfSF["binary_target"], df["Y_pred"]))

    print(classification_report(dfSF["binary_target"], df["Y_pred"]))
    return df

def KMIF_picture(df):
    from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
    plt.rcParams['font.sans-serif'] = ['Simhei']  # 解决中文显示问题，目前只知道黑体可行
    plt.rcParams['axes.unicode_minus'] = False  # 解决负数坐标显示问题
    df = df.iloc[:500, :]

    x3 = df.loc[:, "dst_bytes"][df["Y_pred"] == 0]
    y3 = df.loc[:, "src_bytes"][df["Y_pred"] == 0]
    z3 = df.loc[:, "service"][df["Y_pred"] == 0]

    x4 = df.loc[:, "dst_bytes"][df["Y_pred"] == 1]
    y4 = df.loc[:, "src_bytes"][df["Y_pred"] == 1]
    z4 = df.loc[:, "service"][df["Y_pred"] == 1]

    # 绘制散点图
    fig2 = plt.figure()
    ax2 = Axes3D(fig2)
    ax2.scatter(x3, y3, z3, c='r', label='Normal')
    ax2.scatter(x4, y4, z4, c='g', marker='v', label='Outlier')

    # 绘制图例
    ax2.legend(loc='best')

    # 添加坐标轴(顺序是Z, Y, X)
    ax2.set_zlabel('dst_bytes', fontdict={'size': 15, 'color': 'red'})
    ax2.set_ylabel('src_bytes', fontdict={'size': 15, 'color': 'red'})
    ax2.set_xlabel('service', fontdict={'size': 15, 'color': 'red'})
    # ax2.set_zlim(0, 4100)
    plt.title("KMIF-kddcup scatter plot")
    plt.show()
    plt.savefig('pic/KMIF-KDDCUP异常点分布图.png')
    print(df.columns)

    # return r
if __name__ == "__main__":
    # 1.加载数据集
    target, dfSF = load_dataset()
    print("数据集加载完成。。。")

    # # 2.数据预处理
    dfSF, dfSF_normed = data_process(target, dfSF)
    print("数据集预处理完成。。。")
    ccdata = dfSF_normed

    # # 2.手肘法选择k
    # set_k(ccdata)

    # 3.K-Means进行建模
    df = k_means_model(ccdata, dfSF)
    KMIF_picture(df)
