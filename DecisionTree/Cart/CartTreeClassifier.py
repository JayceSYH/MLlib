from .CartTree import CartTree


class CartTreeClassifier(object):
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
        pred = self.predict(test_df)
        total_num = test_df.shape[0]
        true_num = test_df[test_df[self.label.name] == pred].shape[0]
        return true_num / total_num

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
        :param feature_name:
        :return: sub tree, gini loss
        """

        best_feature_class = None
        max_gini_loss = 0
        for feature_class in feature.class_set:
            gini_loss = self._calculate_discrete_gini(df, feature, feature_class)
            if gini_loss > max_gini_loss:
                max_gini_loss = gini_loss
                best_feature_class = feature_class

        if best_feature_class is None:
            return None, 0

        return self._make_discrete_tree(df, feature, best_feature_class), max_gini_loss

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
        :param feature_name:
        :return: sub tree, gini loss
        """

        sorted_df = df.sort_values(feature.name)
        last_entry_class = None
        max_gini_loss = 0
        best_split_point = None
        last_row = None
        for _, row in sorted_df.iterrows():
            if last_entry_class and row[self.label.name] != last_entry_class:
                split_point = (row[feature.name] + last_row[feature.name]) / 2
                gini_loss = self._calculate_continuous_gini(df, feature, split_point)
                if max_gini_loss < gini_loss:
                    max_gini_loss = gini_loss
                    best_split_point = split_point

            last_row = row
            last_entry_class = row[self.label.name]

        if best_split_point is None:
            return None, 0

        return self._make_continuous_tree(df, feature, best_split_point), max_gini_loss

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
            return self._get_freq_label(df)

        # build tree at current layer
        max_gini_loss = 0
        best_tree = None
        best_feature = None
        for feature in feature_set:
            tree, gini_loss = self._split_feature(df, feature)
            if gini_loss > max_gini_loss:
                best_tree = tree
                max_gini_loss = gini_loss
                best_feature = feature

        if best_tree is None:
            print("no gain")
            return self._get_freq_label(df)

        # recursively build next layer's tree structure
        for node in best_tree.iternodes():
            sub_tree = self._build_tree(node.data, feature_set - {best_feature}, depth + 1, max_depth)
            node.set_sub_tree(sub_tree)

        return best_tree
    
    def _calculate_discrete_gini(self, df, feature, feature_class):
        """
        calculate gini index of discrete feature
        :param df:
        :param feature:
        :param feature_class:
        :return: gini index
        """

        left_part = df[df[feature.name] == feature_class]
        right_part = df[df[feature.name] != feature_class]

        if right_part.shape[0] == 0 or left_part.shape[0] == 0:
            return 0
        
        # calculate gini
        left_gini, right_gini, total_gini = 1, 1, 1
        for label_class in self.label.class_set:
            left_portion = left_part[left_part[self.label.name] == label_class].shape[0] / left_part.shape[0]
            right_portion = right_part[right_part[self.label.name] == label_class].shape[0] / right_part.shape[0]
            total_portion = df[df[self.label.name] == label_class].shape[0] / df.shape[0]

            left_gini -= left_portion ** 2
            right_gini -= right_portion ** 2
            total_gini -= total_portion ** 2

        after_split_gini = float(left_part.shape[0]) / df.shape[0] * left_gini + float(right_part.shape[0]) / df.shape[0] * right_gini

        # calculate gini loss
        return total_gini - after_split_gini

    def _calculate_continuous_gini(self, df, feature, split_point):
        """
        calculate gini index of continuous feature
        :param df:
        :param feature:
        :param split_point:
        :return:
        """

        left_part = df[df[feature.name] <= split_point]
        right_part = df[df[feature.name] > split_point]

        if right_part.shape[0] == 0 or left_part.shape[0] == 0:
            return 0

        # calculate gini
        left_gini, right_gini, total_gini = 1, 1, 1
        for label_class in self.label.class_set:
            left_portion = left_part[left_part[self.label.name] == label_class].shape[0] / left_part.shape[0]
            right_portion = right_part[right_part[self.label.name] == label_class].shape[0] / right_part.shape[0]
            total_portion = df[df[self.label.name] == label_class].shape[0] / df.shape[0]

            left_gini -= left_portion ** 2
            right_gini -= right_portion ** 2
            total_gini -= total_portion ** 2

        after_split_gini = float(left_part.shape[0]) / df.shape[0] * left_gini + float(right_part.shape[0]) / df.shape[0] * right_gini

        # calculate gini loss
        return total_gini - after_split_gini

    def _get_freq_label(self, df):
        """
        get frequentest label
        :param df:
        :return: frequentest label
        """
        max_count = 0
        max_label_class = None
        for label_class in self.label.class_set:
            count = df[df[self.label.name] == label_class].shape[0]
            if count > max_count:
                max_count = count
                max_label_class = label_class

        return max_label_class


class Feature(object):
    def __init__(self, name, feature_type, class_set=None):
        self.name = name
        self.type = feature_type
        self._class_set = class_set

    @property
    def class_set(self):
        assert self.type == "discrete", "Error: continuous feature doesn't have property 'class_set'"
        return self._class_set
