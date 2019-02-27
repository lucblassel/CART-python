#! /usr/bin/env python3

import numpy as np
import pandas as pd
from pandas.api.types import is_number, is_categorical, is_string_dtype, is_bool
from sklearn.datasets import load_iris

def check_categorical(data):
    truth = is_categorical(data) | is_string_dtype(data) | is_bool(data)
    return truth


class Tree:

    def get_indent(level):
        return "    " * level

    def __init__(self, data, outcome, parent_node=None,
                 value="", min_samples_leaf=3, level=0, max_depth=3):
        """
        Arguments:
            data {pd.DataFrame} -- data you want to train the tree on
            outcome {String} -- name of the column containing your target feature
            parent_node {Tree} -- parent node, if node is root then parent_node is None
            value {String} -- value of the of the parent split for this node
            min_samples_leaf {int} -- minimum number of cases in node
        """

        self.data = data
        self.outcome = outcome
        self.parent_node = parent_node
        self.value = value
        self.min_samples_leaf = min_samples_leaf
        self.left = None
        self.right = None
        self.split_on = ""
        self.prediction = None
        self.level = level
        self.max_depth=max_depth

    def display(self):
        lines, _, _, _ = self._display_aux()
        for line in lines:
            print(line)

    def _display_aux(self):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if self.right is None and self.left is None:
            line = '%s' % self.value
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if self.right is None:
            lines, n, p, x = self.left._display_aux()
            s = '%s' % self.value
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if self.left is None:
            lines, n, p, x = self.right._display_aux()
            s = '%s' % self.value
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()
        s = '%s' % self.value
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2

    def get_gini(self, subset):
        """takes the vector of targets as input

        Arguments:
            subset {pd.DataFrame} -- [description]

        Returns:
            [float] -- gini index for given data subset
        """

        proportions = subset[self.outcome].value_counts(normalize=True)
        return 1 - (proportions ** 2).sum()

    def get_delta_i(self, subset):
        gini = self.get_gini(self.data)

        left = subset
        right = self.data.drop(subset.index, axis=0)

        p_left = len(left) / len(self.data)
        p_right = 1 - p_left

        sub_left = p_left * self.get_gini(left)
        sub_right = p_right * self.get_gini(right)

        return gini - sub_left - sub_right

    def get_feature_index(self, feature):
        """gets weighted gini index for a given feature

        Arguments:
            feature {String} -- feature for which to calculate the weighted index

        Returns:
            float -- weighted gini index
        """

        uniques = self.data[feature].value_counts(normalize=True).to_dict()
        ginis = {}
        for unique in uniques:
            gini = self.get_gini(self.data[self.data[feature] == unique])
            ginis[unique] = gini

        return np.sum([ginis[unique] * uniques[unique] for unique in uniques])


    def get_categorical_splits(self, feature):
        splits = {}
        for unique in self.data[feature].unique():
            splits[(feature, unique, 'categorical')] = self.data[
                self.data[feature] == unique]
        return splits

    def get_numerical_splits(self, feature):
        splits = {}
        uniques = self.data[feature].unique()
        for value in uniques:
            if value != max(uniques):
                splits[(feature, value, 'numerical')] = self.data[
                    self.data[feature] <= value]
        return splits

    def get_splits(self):
        features = self.data.columns.drop(self.outcome)
        all_splits = {}

        for feature in features:

            if check_categorical(self.data[feature]):
                all_splits.update(self.get_categorical_splits(feature))
            else:
                all_splits.update(self.get_numerical_splits(feature))

        return all_splits

    # def get_best_split(self):
    #     all_splits = self.get_splits()
    #     ginis = {}

    #     for key, split in all_splits.items():
    #         ginis[key] = self.get_gini(split)

    #     return min(ginis, key=ginis.get)

    def get_best_split(self):
        all_splits = self.get_splits()
        delta_is = {}

        for key, split in all_splits.items():
            delta_is[key] = self.get_delta_i(split)

        return max(delta_is, key=delta_is.get)

    def is_pure(self):
        """Checks if node is pure

        Returns:
            Bool -- purity of node
        """
        pure = len(self.data[self.outcome].unique()) == 1
        small = len(self.data) <= self.min_samples_leaf

        return pure or small

    def too_deep(self):
        return self.level >= self.max_depth

    def no_splits(self):
        return self.get_splits() == {}

    def split(self):
        """Recursive function, that finds the best possible feature to split on in the dataset and creates a child node for each possible value of that feature.
        """

        if self.is_pure() or self.too_deep() or self.no_splits():  # stop condition
            # print(self.data[self.outcome].value_counts(),'\n\n') # debug
            self.prediction = self.data[self.outcome].value_counts().idxmax()
            self.value = ' ({})'.format(
                self.prediction)
            return

        best_split = self.get_best_split()
        self.split_feature = best_split[0]
        self.split_value = best_split[1]
        self.split_type = best_split[2]


        if self.split_type == 'categorical':
            left_data = self.data[
                self.data[self.split_feature] == self.split_value]
            right_data = self.data[
                self.data[self.split_feature] != self.split_value]
            self.value = "{} = {}".format(
                self.split_feature, self.split_value
            )

        elif self.split_type == 'numerical':
            left_data = self.data[
                self.data[self.split_feature] <= self.split_value
            ]
            right_data = self.data[
                self.data[self.split_feature] > self.split_value
            ]
            self.value = "{} <= {}".format(
                self.split_feature, self.split_value
            )
        else:
            raise ValueError(
                'splits can be either numerical or categorical'
                )


        child_params = {
            'outcome': self.outcome,
            'parent_node': self,
            'min_samples_leaf': self.min_samples_leaf,
            'max_depth': self.max_depth,
            'level': self.level +1
        }

        self.left = Tree(left_data, **child_params)
        self.right = Tree(right_data, **child_params)

        # print('split level {}, {}'.format(self.level, self.value))
        # print('left')
        self.left.split()
        # print('right')
        self.right.split()

        return


if __name__ == "__main__":
    data = pd.read_csv('data.tab', sep='\t', header=0, index_col=0)
    data_mixed = pd.read_csv('data_mixed.csv', header=0, index_col=0)
    tree = Tree(data, 'Decision', value='Root')
    tree.split()
    tree.display()

    tree2 = Tree(data_mixed, 'play', value='root')
    tree2.split()
    tree2.display()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1)
    sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue='species')

    ymin = -0.1
    ymax = 2.7
    xmin = 0.7
    xmax = 7.2

    vertical_lines = [
        {'x':1.9,
        'ymin':ymin,
        'ymax':ymax},
        {'x':4.8,
        'ymin':1.7,
        'ymax':ymax},
        {'x':4.9,
        'ymin':ymin,
        'ymax':1.7}
    ]

    horizontal_lines = [
        {'y':1.7,
        'xmin':1.9,
        'xmax':xmax},
        {'y':1.6,
        'xmin':1.9,
        'xmax':4.9},
        {'y':1.5,
        'xmin':4.9,
        'xmax':xmax}
    ]



    for line in horizontal_lines:
        ax.hlines(**line)
    for line in vertical_lines:
        ax.vlines(**line)


