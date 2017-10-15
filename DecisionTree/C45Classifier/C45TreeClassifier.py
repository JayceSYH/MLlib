import math
from functools import reduce
from DecisionTree.C45Classifier.C45Tree import C45Tree


class C45TreeClassifier(object):
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
                if value_num < 50 and value_num < int(df.shape[0] * 0.2):
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
        split discrete feature and return sub tree & entropy gain ratio
        :param df:
        :param feature_name:
        :return: sub tree, entropy gain ratio
        """

        class_dict = self._classify_datas(df, feature)
        info_gain_ratio = self._calculate_discrete_info_gain_ratio(class_dict, df)
        return self._make_discrete_tree(class_dict, df, feature), info_gain_ratio

    def _make_discrete_tree(self, class_dict, total_df, feature):
        """
        make discrete tree
        :param class_dict:
        :param feature:
        :return:
        """

        for k in feature.class_set:
            if k not in class_dict:
                class_dict[k] = self.most_label_class

        most_label_class = None
        biggest_size = 0
        for k in self.label.class_set:
            if total_df[total_df[self.label.name] == k].shape[0] > biggest_size:
                most_label_class = k

        return C45Tree(feature.name, branch_type="discrete", class_dict=class_dict, most_label_class=most_label_class)

    def _split_continuous_feature(self, df, feature):
        """
        split continuous feature and return sub tree & entropy gain ratio
        :param df:
        :param feature_name:
        :return: sub tree, entropy gain ratio
        """

        sorted_df = df.sort_values(feature.name)
        last_entry_class = None
        max_gain = -1
        best_info_gain_ratio = 0
        best_split_point = None
        last_row = None
        for _, row in sorted_df.iterrows():
            if last_entry_class and row[self.label.name] != last_entry_class:
                # 感觉可以优化
                split_point = (row[feature.name] + last_row[feature.name]) / 2
                info_gain, info_gain_ratio = self._calculate_continuous_info_gain_and_ratio(sorted_df, split_point, feature)
                if max_gain < info_gain:
                    max_gain = info_gain
                    best_split_point = split_point
                    best_info_gain_ratio = info_gain_ratio

            last_row = row
            last_entry_class = row[self.label.name]

        assert best_split_point is not None, 'Error：can\' find best split point'
        return self._make_continous_tree(sorted_df, best_split_point, feature), best_info_gain_ratio

    def _make_continous_tree(self, df, split_point, feature):
        """
        make continuous tree
        :param df:
        :param split_point:
        :param feature:
        :return:
        """
        return C45Tree(feature.name, branch_type='continuous', split_point=split_point, class_dict={
            "left":  df[df[feature.name] <= split_point],
            "right": df[df[feature.name] > split_point]
        })

    def _split_feature(self, df, feature):
        """
        split feature and return tree & entropy gain ratio
        :param df:
        :param feature:
        :return: sub tree, entropy gain ratio
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
        max_gain_ratio = 0
        best_tree = None
        best_feature = None
        for feature in feature_set:
            tree, entropy_gain_ratio = self._split_feature(df, feature)
            if entropy_gain_ratio > max_gain_ratio:
                best_tree = tree
                max_gain_ratio = entropy_gain_ratio
                best_feature = feature

        if best_tree is None:
            return self._get_freq_label(df)

        # recursively build next layer's tree structure
        for node in best_tree.iternodes():
            if node.data is not None:
                sub_tree = self._build_tree(node.data, feature_set - {best_feature}, depth + 1, max_depth)
                node.set_sub_tree(sub_tree)

        return best_tree

    def _get_freq_label(self, df):
        max_count = 0
        max_label_class = None
        for label_class in self.label.class_set:
            count = df[df[self.label.name] == label_class].shape[0]
            if count > max_count:
                max_count = count
                max_label_class = label_class

        return max_label_class

    def _classify_datas(self, df, feature):
        """
        classify datas by discrete feature
        :param df:
        :param feature:
        :return: class dictionary
        """
        class_dict = {}
        for feature_value in feature.class_set:
            class_df = df[df[feature.name] == feature_value]
            if class_df.shape[0] > 0:
                class_dict[feature_value] = class_df

        return class_dict

    def _calculate_discrete_info_gain_ratio(self, class_dict, total_df):
        """
        calculate entropy gain ratio of a discrete feature
        :param class_dict:
        :param total_df:
        :return: entropy gain ratio
        """
        total_data_num = total_df.shape[0]
        portion_dict = {}
        entropy_dict = {}

        # calculate entropy after split
        for feature_class in class_dict:
            class_df = class_dict[feature_class]
            class_data_num = class_df.shape[0]
            class_portion_list = self._calculate_portion_list(class_df)
            class_entropy = self._calculate_entropy(class_portion_list)  # 特征每个分类内部的熵
            entropy_dict[feature_class] = class_entropy
            portion_dict[feature_class] = float(class_data_num) / total_data_num

        if len(portion_dict) == 1:
            return 0

        after_split_entropy = 0
        for feature_class in class_dict:
            after_split_entropy += portion_dict[feature_class] * entropy_dict[feature_class]

        # calculate split's entropy
        sub_tree_entropy = self._calculate_entropy(portion_dict)

        # calculate entropy before split
        before_split_portion_list = self._calculate_portion_list(total_df)
        before_split_entropy = self._calculate_entropy(before_split_portion_list)

        if sub_tree_entropy == 0:
            print()

        return (before_split_entropy - after_split_entropy) / sub_tree_entropy

    def _calculate_portion_list(self, df):
        """
        calculate label class portion of source data
        :param df:
        :return: portion list
        """
        total_data_num = df.shape[0]
        portion_list = []
        for label_class in self.label.class_set:
            portion_list.append(df[df[self.label.name] == label_class].shape[0] / total_data_num)

        return portion_list

    def _calculate_entropy(self, portions):
        """
        calculate entropy given a list or a dict of probabilities
        :param portions: 
        :return: entropy
        """
        if isinstance(portions, dict):
            portions = [portions[k] for k in portions]

        portions = [v for v in portions if v > 0]

        entropy = reduce(lambda x, y: x + y, 
                         map(lambda x: -x * math.log2(x), portions))
        
        return entropy

    def _calculate_continuous_info_gain_and_ratio(self, df, split_point, feature):
        """
        calculate info gain and info gain ratio
        :param df:
        :param split_point:
        :param feature:
        :return: info gain, info gain ratio
        """
        total_data_num = df.shape[0]
        
        left_part = df[df[feature.name] <= split_point]
        right_part = df[df[feature.name] > split_point]

        if left_part.shape[0] == 0 or right_part.shape[0] == 0:
            return 0, 0
        
        # calculate entropy after split
        left_portion_list = self._calculate_portion_list(left_part)
        left_entropy = self._calculate_entropy(left_portion_list)
        left_portion = left_part.shape[0] / total_data_num

        right_portion_list = self._calculate_portion_list(right_part)
        right_entropy = self._calculate_entropy(right_portion_list)
        right_portion = right_part.shape[0] / total_data_num

        after_split_entropy = left_portion * left_entropy + right_portion * right_entropy

        # calculate split's entropy
        sub_tree_entropy = self._calculate_entropy([left_portion, right_portion])

        # calculate entropy before split
        before_split_portion_list = self._calculate_portion_list(df)
        before_split_entropy = self._calculate_entropy(before_split_portion_list)

        # calculate info gain and info gain ratio
        info_gain = before_split_entropy - after_split_entropy
        info_gain_ratio = info_gain / sub_tree_entropy

        return info_gain, info_gain_ratio


class Feature(object):
    def __init__(self, name, feature_type, class_set=None):
        self.name = name
        self.type = feature_type
        self._class_set = class_set

    @property
    def class_set(self):
        assert self.type == "discrete", "Error: continuous feature doesn't have property 'class_set'"
        return self._class_set
