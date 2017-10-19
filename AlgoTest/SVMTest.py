from SVM.SVMClassifier import SVMClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def NTestSVM():
    total_df = pd.read_csv("iris.csv")
    total_num = total_df.shape[0]
    total_df = total_df.sample(frac=1)
    train_df = total_df.iloc[:int(total_num * 0.8)]
    test_df = total_df.iloc[int(total_num * 0.8):]
    svm = SVMClassifier()
    svm.fit(train_df.drop(['species'], axis=1).as_matrix(),
            train_df.apply(lambda row: 1 if row['species'] == "setosa" else -1, axis=1))
    pred = svm.predict(test_df.drop(['species'], axis=1).as_matrix())
    truth = test_df.apply(lambda row: 1 if row['species'] == "setosa" else -1, axis=1)
    comp_list = list(map(lambda tp: 1 if tp[0] == tp[1] else 0, zip(pred, truth)))
    score = np.mean(comp_list)
    print(score)

def NTest2():
    data = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=1000)
    y = np.array(list(map(lambda tp: 1 if tp[0] * 2 > tp[1] else -1, data)))
    a1 = data[y == 1]
    a2 = data[y == -1]
    svm = SVMClassifier(C=10)
    svm.fit(data[:800], y[:800])
    pred = svm.predict(data[800:])
    truth = y[800:]

    comp_list = list(map(lambda tp: 1 if tp[0] == tp[1] else 0, zip(pred, truth)))
    score = np.mean(comp_list)
    print(score)

    plt.scatter(a1[:, 0], a1[:, 1], c='b', s=1)
    plt.scatter(a2[:, 0], a2[:, 1], c='r', s=1)
    x = np.linspace(-2, 2, 1000)
    y = list(map(lambda x: (-svm.b - svm.w[0] * x) / svm.w[1], x))
    plt.scatter(x, y, s=1)
    plt.show()

def NTestsklearn():
    from sklearn.svm import SVC
    data = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=1000)
    y = np.array(list(map(lambda tp: 1 if tp[0] * 2 > tp[1] else -1, data)))
    svm = SVC()
    svm.fit(data[:800], y[:800])
    print(svm.score(data[800:], y[800:]))



if __name__ == "__main__":
    NTest2()