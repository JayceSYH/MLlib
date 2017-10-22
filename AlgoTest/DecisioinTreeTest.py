import pandas as pd
from DecisionTree.C45Classifier import C45TreeClassifier
from DecisionTree.Cart import CartTreeClassifier
from DecisionTree.Cart import CartTreeRegressor


def NTestC45():
    total_df = pd.read_csv("iris.csv").sample(frac=1)
    total_num = total_df.shape[0]
    train_df = total_df.iloc[:int(total_num * 0.8)]
    test_df = total_df.iloc[int(total_num * 0.8):]
    c45 = C45TreeClassifier()
    c45.fit(train_df, "species")
    print(c45.score(test_df))

def NTestCart():
    total_df = pd.read_csv("iris.csv").sample(frac=1)
    total_num = total_df.shape[0]
    train_df = total_df.iloc[:int(total_num * 0.8)]
    test_df = total_df.iloc[int(total_num * 0.8):]
    cart = CartTreeClassifier()
    cart.fit(train_df, "species")
    print(cart.score(test_df))


def NTestC45_titanic():
    total_df = pd.read_csv("titanic_clean.csv")
    total_df.drop(['cabin', 'boat', 'body', 'index'], axis=1, inplace=True)
    total_df.dropna(inplace=True)
    total_num = total_df.shape[0]
    train_df = total_df.iloc[:int(total_num * 0.8)]
    test_df = total_df.iloc[int(total_num * 0.8):]
    c45 = C45TreeClassifier()
    c45.fit(train_df, "survived")
    print(c45.score(test_df))

def NTestCart_titanic():
    total_df = pd.read_csv("titanic_clean.csv")
    total_df.drop(['cabin', 'boat', 'body', 'index'], axis=1, inplace=True)
    total_df.dropna(inplace=True)
    total_num = total_df.shape[0]
    train_df = total_df.iloc[:int(total_num * 0.8)]
    test_df = total_df.iloc[int(total_num * 0.8):]
    cart = CartTreeClassifier()
    cart.fit(train_df, "survived")
    print(cart.score(test_df))

def NTestCartRegression_titanic():
    total_df = pd.read_csv("titanic_clean.csv")
    total_df.drop(['cabin', 'boat', 'body', 'index'], axis=1, inplace=True)
    total_df.dropna(inplace=True)
    total_num = total_df.shape[0]
    train_df = total_df.iloc[:int(total_num * 0.8)]
    test_df = total_df.iloc[int(total_num * 0.8):]
    cart = CartTreeRegressor()
    cart.fit(train_df, "fare")
    print(cart.score(test_df))

def sklearn_titanic():
    from sklearn.tree.tree import DecisionTreeClassifier
    from sklearn.preprocessing.label import LabelEncoder
    total_df = pd.read_csv("titanic_clean.csv")
    total_df.drop(['cabin', 'boat', 'body', 'index'], axis=1, inplace=True)
    total_df.dropna(inplace=True)
    for col in total_df.columns.tolist():
        if str(total_df[col].dtype) == 'object':
            total_df[col] = LabelEncoder().fit_transform(total_df[col])

    total_num = total_df.shape[0]
    train_df = total_df.iloc[:int(total_num * 0.8)]
    test_df = total_df.iloc[int(total_num * 0.8):]

    clf = DecisionTreeClassifier()
    clf.fit(train_df.drop(['survived'], axis=1), train_df['survived'])
    print(clf.score(test_df.drop(['survived'], axis=1), test_df['survived']))


def sklearn_titanic_regression():
    from sklearn.tree.tree import DecisionTreeRegressor
    from sklearn.preprocessing.label import LabelEncoder
    import numpy as np
    total_df = pd.read_csv("titanic_clean.csv")
    total_df.drop(['cabin', 'boat', 'body', 'index'], axis=1, inplace=True)
    total_df.dropna(inplace=True)
    for col in total_df.columns.tolist():
        if str(total_df[col].dtype) == 'object':
            total_df[col] = LabelEncoder().fit_transform(total_df[col])

    total_num = total_df.shape[0]
    train_df = total_df.iloc[:int(total_num * 0.8)]
    test_df = total_df.iloc[int(total_num * 0.8):]

    clf = DecisionTreeRegressor()
    clf.fit(train_df.drop(['fare'], axis=1), train_df['fare'])
    pred = clf.predict(test_df.drop(['fare'], axis=1))
    truth = test_df['fare']
    mse = np.sum(np.square(pred - truth)) / test_df.shape[0]
    print(mse)


def NTestNormCart():
    import numpy as np
    from sklearn.svm import SVC
    data = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=1000)
    y = np.array(list(map(lambda tp: 1 if tp[0] * 2 > tp[1] else -1, data)))
    data = pd.DataFrame(data)
    data[2] = y
    cart = CartTreeClassifier()
    cart.fit(data[:800], 2)
    print(cart.score(data[800:]))


if __name__ == "__main__":
    sklearn_titanic_regression()