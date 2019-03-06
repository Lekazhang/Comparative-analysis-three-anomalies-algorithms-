import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn
import scipy.io as scio




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

    # 计算真实异常率
    print('Fradulent Transaction:', count_fraud_trans)
    print('Valid Transactions:', count_valid_trans)
    print('Percentage outlier: ', percent_outlier)
    print("SF Anomaly Rate is:" + "{:.1%}".format(percent_outlier))  # 异常率53.8%
    return ccdata_new, percent_outlier

def explore_analysis(ccdata):
    import seaborn as sns
    correlation_matrix = ccdata.corr()
    fig = plt.figure(figsize=(12, 9))

    sns.heatmap(correlation_matrix, vmax=1, square=True)
    plt.show()

def data_process(ccdata):
    # 将数据集拆分为训练集 == > 所有参数（或仅相关参数）
    # 即训练和测试集，以及评估列Class。
    # 从数据框中提取列。

    columns = ccdata.columns.tolist()

    # 根据需要过滤列
    # 1。所有V参数和排除类
    columns_V_all = [c for c in columns if c not in ["Class"]]


    col_eval = ccdata["Class"]
    return columns_V_all, col_eval

def model(ccdata, columns_V_all, percent_outlier):
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn import svm
    import datetime

    X_types = [columns_V_all]
    percent_outlier = 0.5
    print(len(ccdata))
    for x in X_types:

        models = {"LOF": LocalOutlierFactor(n_neighbors=1, contamination=0.5),
                  "IsF": IsolationForest(max_samples=len(ccdata[x]), contamination=0.4, random_state=1),
                  "OCS": svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=0.1)}
        keys = ["LOF", "IsF", "OCS"]

        if keys[0] == "LOF":
            mod_name = "LOF"
            model = models.get("LOF")
            # LOF计算运行时间
            start1 = datetime.datetime.now()
            Y_pred = model.fit_predict(ccdata[x])
            scores_pred = model.negative_outlier_factor_
            end1 = datetime.datetime.now()
            print("LOF 运行时间：", end1 - start1)

            # 默认情况下，这些模型的预测值给出-1和+1，需要将其更改为0和1
            Y_pred[Y_pred == 1] = 0
            Y_pred[Y_pred == -1] = 1
            error_count = (Y_pred != col_eval).sum()
            # Printing the metrics for the classification algorithms
            print('{}: Number  of errors {}'.format(mod_name, error_count))
            print("accuracy score: ", accuracy_score(col_eval, Y_pred))
            print(classification_report(col_eval, Y_pred))

        if keys[1] == "IsF":
            mod_name = "IsF"
            model = models.get("IsF")
            # IF计算运行时间
            start2 = datetime.datetime.now()
            model.fit(ccdata[x])
            scores_pred = model.decision_function(ccdata[x])
            Y_pred = model.predict(ccdata[x])
            end2 = datetime.datetime.now()
            print("IF 运行时间：", end2 - start2)
            # 默认情况下，这些模型的预测值给出-1和+1，需要将其更改为0和1
            Y_pred[Y_pred == 1] = 0
            Y_pred[Y_pred == -1] = 1
            error_count = (Y_pred != col_eval).sum()
            # Printing the metrics for the classification algorithms
            print('{}: Number  of errors {}'.format(mod_name, error_count))
            print("accuracy score: ", accuracy_score(col_eval, Y_pred))
            print(classification_report(col_eval, Y_pred))

        if keys[2] == "OCS":
            mod_name = "OCS"
            model = models.get("OCS")
            # OCS计算运行时间
            start3 = datetime.datetime.now()
            model.fit(ccdata[x])
            Y_pred = model.predict(ccdata[x])
            end3 = datetime.datetime.now()
            print("OCS 运行时间：", end3 - start3)

            # 默认情况下，这些模型的预测值给出-1和+1，需要将其更改为0和1
            Y_pred[Y_pred == 1] = 0
            Y_pred[Y_pred == -1] = 1
            error_count = (Y_pred != col_eval).sum()
            # Printing the metrics for the classification algorithms
            print('{}: Number  of errors {}'.format(mod_name, error_count))
            print("accuracy score: ", accuracy_score(col_eval, Y_pred))
            print(classification_report(col_eval, Y_pred))



if __name__ == "__main__":
    # 1.加载数据集
    ccdata, percent_outlier = load_dataset()
    print("数据集加载完成。。。")

    # 2.探索性分析
    # explore_analysis(ccdata)

    # 3.数据预处理
    columns_V_all, col_eval= data_process(ccdata)
    print("数据集预处理完成。。。")

    # 4.监督环境下对LOF,IF建模
    model(ccdata, columns_V_all, percent_outlier)