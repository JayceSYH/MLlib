import pandas as pd
from DecisionTree.C45Classifier import C45TreeClassifier


def NTestC45():
    total_df = pd.read_csv("iris.csv")
    total_num = total_df.shape[0]
    train_df = total_df.iloc[:int(total_num * 0.8)]
    test_df = total_df.iloc[int(total_num * 0.8):]
    c45 = C45TreeClassifier()
    c45.fit(train_df, "species")
    print(c45.score(test_df))


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

if __name__ == "__main__":
    NTestC45_titanic()