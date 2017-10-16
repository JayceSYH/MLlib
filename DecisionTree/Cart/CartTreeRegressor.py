from .CartTree import CartTree
import numpy as np


class CartTreeRegressor(object):
    def __init__(self, max_depth=16):
        self.max_depth = max_depth
        self.feature_set = set()
        self.tree = None
        self.label = None
        self.most_label_class = None

    def fit(self, train_df, label_name):
        """
        fit training data
        :param train_df:
        :param label_name:
        :return: None
        """
        self.preprocess(train_df, label_name)
        self.tree = self._build_tree(train_df, self.feature_set, depth=1, max_depth=self.max_depth)

    def predict(self, df):
        """
        predict target class value
        :param df:
        :return: predict value
        """
        return df.apply(lambda row: self.tree.predict(row), axis=1)

    def score(self, test_df):
        """
        score test data
        :param test_df:
        :return: score(max 1)
        """
        pred = np.array(self.predict(test_df))
        truth = np.array(test_df[self.label.name])
        return np.sum(np.square(pred - truth)) / test_df.shape[0]

    def preprocess(self, df, label_name):
        """
        preprocess training data
        :param df:
        :param label_name:
        :return: None
        """
        feature_names = df.columns.tolist()
        if label_name not in feature_names:
            raise Exception("no column named '{}' found".format(label_name))
        for feature_name in feature_names:
            if feature_name == label_name:
                continue
            dtype = str(df[feature_name].dtype)
            if "float" in dtype or "int" in dtype:
                feature_classes = set(df[feature_name])
                value_num = len(feature_classes)
                if value_num < 10 and value_num < int(df.shape[0] * 0.2):
                    self.feature_set.add(Feature(feature_name, "discrete", feature_classes))
                else:
                    self.feature_set.add(Feature(feature_name, "continuous", feature_classes))
            else:
                self.feature_set.add(Feature(feature_name, "discrete", set(df[feature_name])))

        self.label = Feature(label_name, "discrete", set(df[label_name]))

        biggest_size = 0
        for label_class in self.label.class_set:
            size = df[df[label_name] == label_class].shape[0]
            if size > biggest_size:
                self.most_label_class = label_class

    def _split_discrete_feature(self, df, feature):
        """
        split discrete feature and return sub tree & gini index
        :param df:
        :param feature:
        :return: sub tree, gini loss
        """

        best_feature_class = None
        min_mse = None
        for feature_class in feature.class_set:
            mse = self._calculate_discrete_mse(df, feature, feature_class)
            if mse is not None and (min_mse is None or mse < min_mse):
                min_mse = mse
                best_feature_class = feature_class

        if best_feature_class is None:
            return None, None

        return self._make_discrete_tree(df, feature, best_feature_class), min_mse

    def _make_discrete_tree(self, df, feature, best_feature_class):
        """
        make discrete tree
        :param df:
        :param feature:
        :param best_feature_class:
        :return:
        """

        return CartTree(feature.name, best_feature_class, df, "discrete")

    def _split_continuous_feature(self, df, feature):
        """
        split discrete feature and return sub tree & gini index
        :param df:
        :param feature:
        :return: sub tree, mse
        """

        sorted_df = df.sort_values(feature.name)
        min_mse = None
        best_split_point = None
        for _, row in sorted_df.iterrows():
            split_point = row[feature.name]
            mse = self._calculate_continuous_mse(df, feature, split_point)
            if mse is not None and (min_mse is None or min_mse > mse):
                min_mse = mse
                best_split_point = split_point

        if best_split_point is None:
            return None, None

        return self._make_continuous_tree(df, feature, best_split_point), min_mse

    def _make_continuous_tree(self, df, feature, best_split_point):
        """
        make continuous tree
        :param df:
        :param feature:
        :param best_split_point:
        :return: Cart Tree
        """

        return CartTree(feature.name, best_split_point, df, "continuous")

    def _split_feature(self, df, feature):
        """
        split feature and return tree & gini index
        :param df:
        :param feature:
        :return: sub tree, gini loss
        """
        if feature.type == "discrete":
            return self._split_discrete_feature(df, feature)
        else:
            return self._split_continuous_feature(df, feature)

    def _build_tree(self, df, feature_set, depth, max_depth):
        """
        recursively build decision tree
        :param df:
        :param feature_set:
        :return: tree
        """

        # left label classes
        left_classes = set(df[self.label.name])

        # if depth is bigger than max depth or there's only one kind of label left or there's no more
        # feature to make trees, return label
        if depth > max_depth or len(left_classes) == 1 or len(feature_set) == 0:
            return self._get_mean(df)

        # build tree at current layer
        min_mse = None
        best_tree = None
        best_feature = None
        for feature in feature_set:
            tree, mse = self._split_feature(df, feature)
            if mse is not None and (min_mse is None or min_mse > mse):
                best_tree = tree
                min_mse = mse
                best_feature = feature

        if best_tree is None:
            print("no gain")
            return self._get_mean(df)

        # recursively build next layer's tree structure
        for node in best_tree.iternodes():
            sub_tree = self._build_tree(node.data, feature_set - {best_feature}, depth + 1, max_depth)
            node.set_sub_tree(sub_tree)

        return best_tree

    def _calculate_discrete_mse(self, df, feature, feature_class):
        """
        calculate mse of discrete feature
        :param df:
        :param feature:
        :param feature_class:
        :return: mse
        """

        left_part = df[df[feature.name] == feature_class]
        right_part = df[df[feature.name] != feature_class]

        if right_part.shape[0] == 0 or left_part.shape[0] == 0:
            return None

        # calculate mse
        left_mse = left_part[self.label.name].var() * left_part.shape[0]
        right_mse = right_part[self.label.name].var() * right_part.shape[0]

        return left_mse + right_mse

    def _calculate_continuous_mse(self, df, feature, split_point):
        """
        calculate mse of discrete feature
        :param df:
        :param feature:
        :param split_point:
        :return: mse
        """

        left_part = df[df[feature.name] <= split_point]
        right_part = df[df[feature.name] > split_point]

        if right_part.shape[0] == 0 or left_part.shape[0] == 0:
            return None

        # calculate mse
        left_mse = left_part[self.label.name].var() * left_part.shape[0]
        right_mse = right_part[self.label.name].var() * right_part.shape[0]

        return left_mse + right_mse

    def _get_mean(self, df):
        """
        get mean of target value
        :param df:
        :return: target mean
        """
        return df[self.label.name].mean()


class Feature(object):
    def __init__(self, name, feature_type, class_set=None):
        self.name = name
        self.type = feature_type
        self._class_set = class_set

    @property
    def class_set(self):
        assert self.type == "discrete", "Error: continuous feature doesn't have property 'class_set'"
        return self._class_set