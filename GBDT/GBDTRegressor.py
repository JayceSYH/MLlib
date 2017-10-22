from DecisionTree import CartTreeRegressor
import numpy as np


class GBDTRegressor(object):
    def __init__(self, n_estimator=10):
        self.n_estimator = n_estimator
        self.trees = []
        self.label_name = None

    def fit(self, df, label_name):
        self.label_name = label_name
        df_train = df.drop([label_name], axis=1)
        df_train['__GBDT_RESIDUAL__'] = df[label_name]

        for i in range(self.n_estimator):
            cart = CartTreeRegressor(max_depth=8)
            cart.fit(df_train, '__GBDT_RESIDUAL__')
            pred = cart.predict(df_train)
            df_train['__GBDT_RESIDUAL__'] = df_train['__GBDT_RESIDUAL__'] - pred
            self.trees.append(cart)

    def predict(self, df):
        total_pred = None
        for tree in self.trees:
            pred = tree.predict(df)
            if total_pred is None:
                total_pred = pred
            else:
                total_pred += pred

        return total_pred

    def score(self, df):
        pred = np.array(self.predict(df))
        truth = np.array(df[self.label_name])
        return np.sum(np.square(pred - truth)) / df.shape[0]