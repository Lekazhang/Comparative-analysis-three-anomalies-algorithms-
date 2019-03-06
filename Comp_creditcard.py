import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn




def load_dataset():
    ccdata = pd.read_csv('duibi_dataset/creditcard.csv', low_memory=False)

    # print(ccdata.columns)
    # print(ccdata.shape)
    # print(ccdata.describe)

    count_fraud_trans = ccdata['Class'][ccdata['Class'] == 1].count()
    count_valid_trans = ccdata['Class'][ccdata['Class'] == 0].count()
    percent_outlier = count_fraud_trans / (count_valid_trans)

    # 计算真实异常率
    print('Fradulent Transaction:', count_fraud_trans)
    print('Valid Transactions:', count_valid_trans)
    print('Percentage outlier: ', percent_outlier)
    print("SF Anomaly Rate is:" + "{:.1%}".format(percent_outlier))
    return ccdata,percent_outlier

def explore_analysis(ccdata):
    import seaborn as sns
    correlation_matrix = ccdata.corr()
    fig = plt.figure(figsize=(12, 9))

    sns.heatmap(correlation_matrix, vmax=0.4, square=True)
    plt.show()

def data_process(ccdata):
    # 将数据集拆分为训练集 == > 所有参数（或仅相关参数）
    # 即训练和测试集，以及评估列Class。
    # 从数据框中提取列。

    columns = ccdata.columns.tolist()

    # 根据需要过滤列
    # 1。所有V参数和排除类
    columns_V_all = [c for c in columns if c not in ["Class"]]

    # 2. 一些与Class相关的V参数，不包括class和amount列
    columns_V_part = [c for c in columns_V_all if
                      c not in ["Class", "Amount", "V22", "V23", "V24", "V25", "V26", "V27", "V28"]]

    col_eval = ccdata["Class"]
    return columns_V_all, columns_V_part, col_eval

def model(ccdata, columns_V_all, columns_V_part, percent_outlier):
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn import svm
    import datetime

    X_types = [columns_V_part]

    for x in X_types:

        # models = {"LOF": LocalOutlierFactor(n_neighbors=20, contamination=percent_outlier),
        #           "IsF": IsolationForest(max_samples=len(ccdata[x]), contamination=percent_outlier, random_state=1),
        #           "OCS": svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)}

        models = {"LOF": LocalOutlierFactor(n_neighbors=20, contamination=0.3),
                  "IsF": IsolationForest(max_samples=256, contamination=0.1, random_state=1, n_estimators=200),
                  "OCS": svm.OneClassSVM(nu=0.2, kernel="rbf", gamma=0.1)}

        print(len(x))
        keys = ["LOF", "IsF", "OCS"]

        # if keys[0] == "LOF":
        #     mod_name = "LOF"
        #     model = models.get("LOF")
        #     # LOF计算运行时间
        #     start1 = datetime.datetime.now()
        #     Y_pred = model.fit_predict(ccdata[x])
        #     scores_pred = model.negative_outlier_factor_
        #     end1 = datetime.datetime.now()
        #     print("LOF 运行时间：", end1 - start1)
        #
        #     # 默认情况下，这些模型的预测值给出-1和+1，需要将其更改为0和1
        #     Y_pred[Y_pred == 1] = 0
        #     Y_pred[Y_pred == -1] = 1
        #     error_count = (Y_pred != col_eval).sum()
        #     # Printing the metrics for the classification algorithms
        #     print('{}: Number  of errors {}'.format(mod_name, error_count))
        #     print("accuracy score: ", accuracy_score(col_eval, Y_pred))
        #     print(classification_report(col_eval, Y_pred))

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

    # # # 2.探索性分析
    # explore_analysis(ccdata)
    #
    # 3.数据预处理
    columns_V_all, columns_V_part, col_eval= data_process(ccdata)
    print("数据集预处理完成。。。")

    # 4.监督环境下对LOF,IF建模
    model(ccdata, columns_V_all, columns_V_part, percent_outlier)