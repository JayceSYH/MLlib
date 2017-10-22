import pandas as pd
from GBDT.GBDTRegressor import GBDTRegressor


def NTesGBDTRegression_titanic():
    total_df = pd.read_csv("titanic_clean.csv")
    total_df.drop(['cabin', 'boat', 'body', 'index'], axis=1, inplace=True)
    total_df.dropna(inplace=True)
    total_num = total_df.shape[0]
    train_df = total_df.iloc[:int(total_num * 0.8)]
    test_df = total_df.iloc[int(total_num * 0.8):]
    cart = GBDTRegressor(n_estimator=2)
    cart.fit(train_df, "fare")
    print(cart.score(test_df))

if __name__ == "__main__":
    NTesGBDTRegression_titanic()