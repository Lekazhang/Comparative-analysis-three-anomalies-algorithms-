from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn import svm
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn import metrics

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import datetime


def byte_decoder(val):
    # decodes byte literals to strings

    return val.decode('utf-8')


def plot_confusion_matrix(cm, title, classes=['abnormal', 'normal'],
                          cmap=plt.cm.Blues, save=False, saveas="MyFigure.png"):
    # print Confusion matrix with blue gradient colours

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.1%'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if save:
        plt.savefig(saveas, dpi=100)


def plot_gridsearch_cv(results, estimator, x_min, x_max, y_min, y_max, save=False, saveas="MyFigure.png"):
    # print GridSearch cross-validation for parameters
    scoring = {'AUC': 'roc_auc', 'Recall': make_scorer(recall_score, pos_label=-1)}
    plt.figure(figsize=(10, 8))
    plt.title("GridSearchCV for " + estimator, fontsize=24)

    plt.xlabel(estimator)
    plt.ylabel("Score")
    plt.grid()

    ax = plt.axes()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    pad = 0.005
    X_axis = np.array(results["param_" + estimator].data, dtype=float)

    for scorer, color in zip(sorted(scoring), ['b', 'k']):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                    (X_axis[best_index], best_score + pad))

    plt.legend(loc="best")
    plt.grid('off')
    plt.tight_layout()
    if save:
        plt.savefig(saveas, dpi=100)

    plt.show()


def load_dataset():
    target = 'target'
    sf = datasets.fetch_kddcup99(subset='SF', percent10=False)
    dfSF = pd.DataFrame(sf.data,
                        columns=["duration", "service", "src_bytes", "dst_bytes"])
    assert len(dfSF) > 0, "SF dataset no loaded."

    dfSF[target] = sf.target
    anomaly_rateSF = 1.0 - len(dfSF.loc[dfSF[target] == b'normal.']) / len(dfSF)

    # 计算数据集数量
    print("kddcup长度：", len(dfSF))
    # 计算真实异常率
    print("SF Anomaly Rate is:" + "{:.1%}".format(anomaly_rateSF))
    return target,dfSF

def data_process(target,dfSF):
    # 将热编码应用于string类型的字段
    # 将所有异常目标类型转换为单个异常类
    toDecodeSF = ['service', target]

    dfSF['binary_target'] = [1 if x == b'normal.' else -1 for x in dfSF[target]]

    leSF = preprocessing.LabelEncoder()

    for f in toDecodeSF:
        dfSF[f] = list(map(byte_decoder, dfSF[f]))
        dfSF[f] = leSF.fit_transform(dfSF[f])

    dfSF_normed = preprocessing.normalize(dfSF.drop([target, 'binary_target'], axis=1))
    return dfSF,dfSF_normed

def model(dfSF,dfSF_normed):
    X_train_sf, X_test_sf, y_train_sf, y_test_sf = train_test_split(dfSF.drop([target, 'binary_target'], axis=1),
                                                                    dfSF['binary_target'], test_size=0.33,
                                                                    random_state=11)

    X_train_nd, X_test_nd, y_train_nd, y_test_nd = train_test_split(dfSF_normed, dfSF['binary_target'],
                                                                    test_size=0.33, random_state=11)




    clfIF = IsolationForest(max_samples=len(X_train_sf), contamination=0.1, n_estimators=100, n_jobs=-1, behaviour="new")
    # clfLOF = LocalOutlierFactor(n_neighbors=15, metric='euclidean', algorithm='auto', contamination=0.15, n_jobs=-1)
    clfLOF = LocalOutlierFactor(n_neighbors=20, contamination=0.15)

    # 计算IF运行时间
    start1 = datetime.datetime.now()

    clfIF.fit(X_train_sf, y_train_sf)
    y_pred_train = clfIF.predict(X_train_sf)
    end1 = datetime.datetime.now()
    print(end1 - start1)

    # 计算LOF运行时间
    start2 = datetime.datetime.now()
    y_pred_train_lof = clfLOF.fit_predict(X_train_nd, y_train_nd)
    end2 = datetime.datetime.now()
    print(end2 - start2)


    return X_train_sf, X_test_sf, y_train_sf, y_test_sf, X_train_nd, X_test_nd, y_train_nd, y_test_nd, y_pred_train, y_pred_train_lof

def display_pic_IF(X_train_sf, y_train_sf,y_pred_train):

    #
    print("accuracy score: ", accuracy_score(y_train_sf, y_pred_train))
    print(classification_report(y_train_sf, y_pred_train, target_names=['anomaly', 'normal']))

    # print("AUC: ", "{:.1%}".format(roc_auc_score(y_train_sf, y_pred_train)))
    # cm = confusion_matrix(y_train_sf, y_pred_train)
    # plot_confusion_matrix(cm, title="IF Confusion Matrix - SF", save=True, saveas="IF_SF.png")
    # plt.show()

def display_pic_LOF(X_train_nd, y_train_nd,y_pred_train_lof):

    # LOF可视化结果展示
    print("accuracy score: ", accuracy_score(y_train_nd, y_pred_train_lof))
    print(classification_report(y_train_nd, y_pred_train_lof, target_names=['anomaly', 'normal']))

    # print("AUC: ", "{:.1%}".format(roc_auc_score(y_train_nd, y_pred_train_lof)))
    # cm = confusion_matrix(y_train_nd, y_pred_train_lof)
    # plot_confusion_matrix(cm, title="LOF Confusion Matrix - SF")
    # plt.show()

def display_pic_OCS(y_train_sv,y_pred_train_ocs):

    # LOF可视化结果展示
    print("accuracy score: ", accuracy_score(y_train_sv, y_pred_train_ocs))
    print(classification_report(y_train_sv, y_pred_train_ocs, target_names=['anomaly', 'normal']))


if __name__ == "__main__":
    # 1.加载数据集
    target,dfSF = load_dataset()
    print("数据集加载完成。。。")

    # # 2.数据预处理
    dfSF, dfSF_normed = data_process(target,dfSF)
    print("数据集预处理完成。。。")

    # 3.监督环境下对LOF,IF建模
    X_train_sf, X_test_sf, y_train_sf, y_test_sf, X_train_nd, X_test_nd, y_train_nd, y_test_nd, y_pred_train, y_pred_train_lof= model(dfSF,dfSF_normed)
    print("数据建模完成。。。")

    # 4.可视化结果展示
    display_pic_IF(X_train_sf, y_train_sf, y_pred_train)
    # display_pic_LOF(X_train_nd, y_train_nd,y_pred_train_lof)
    print("可视化结果保存完成。。。")


