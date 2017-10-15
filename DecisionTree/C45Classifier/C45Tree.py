import pandas as pd


class C45Tree(object):
    def __init__(self, feature_name, branch_type="discrete", class_dict=None, split_point=None, most_label_class=None):
        self._type = branch_type
        self._feature_name = feature_name
        self._split_point = split_point
        self._nodes = {}
        if branch_type == "discrete":
            assert class_dict is not None and most_label_class is not None, \
                "Error：discrete feature must have 'class_dict' and 'most_label_class' param"
            self._most_label_class = most_label_class
            for feature_class in class_dict:
                data = class_dict[feature_class]
                if isinstance(data, pd.DataFrame):
                    self._nodes[feature_class] = C45TreeNode(data=data, feature_class=feature_class)
                else:
                    self._nodes[feature_class] = C45TreeNode(data=None, feature_class=feature_class, sub_tree=data)
        else:
            assert split_point is not None and class_dict is not None, \
                "Error：continuous feature must have 'split_point' and 'class_dict' param"
            self._nodes['left'] = C45TreeNode(data=class_dict['left'])
            self._nodes['right'] = C45TreeNode(data=class_dict['right'])

    def iternodes(self):
        return [self._nodes[k] for k in self._nodes]

    def clean_data(self):
        for node in self._nodes:
            node.clean_data()

    def predict(self, data_entry):
        if self._type == "discrete":
            if data_entry[self._feature_name] in self._nodes:
                sub_tree = self._nodes[data_entry[self._feature_name]].sub_tree
            else:
                sub_tree = self._most_label_class
            if isinstance(sub_tree, C45Tree):
                print("{0}({1})=>".format(self._feature_name, data_entry[self._feature_name]), end="")
                return sub_tree.predict(data_entry)
            else:
                print("{0}({1})=>{2}".format(self._feature_name, data_entry[self._feature_name], sub_tree))
                return sub_tree

        else:
            sub_tree = self._nodes["left" if data_entry[self._feature_name] < self._split_point else "right"].sub_tree
            if isinstance(sub_tree, C45Tree):
                return sub_tree.predict(data_entry)
            else:
                return sub_tree


class C45TreeNode(object):
    def __init__(self, data=None, feature_class=None, sub_tree=None):
        self.sub_tree = sub_tree
        self.data = data
        self.feature_class = feature_class

    def set_sub_tree(self, tree):
        self.sub_tree = tree

    def clean_data(self):
        self.data = None