"""
Implement decision tree

tested in Python 3.6 :: Miniconda
"""

import argparse
import random
import copy
import numpy as np

import help_functions


class Node():
    """
    Basic node in the decision tree
    """
    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.feature = None
        self.feature_split_boundary = None
        self.label = None


def majority_voting(labels):
    """
    Get the majority voting for a list of labels

    Parameters
    --------------
    labels : list
        a list of classfied labels for a single record

    Returns
    --------------
    label got from majority voting.
    """
    return max(set(labels), key=lambda label: labels.count(label))


def gini(labels):
    """
    Calculate the gini index for a single node

    Parameters
    -------------
    labels : list
        list of labels

    Returns
    ---------------
    gini_index : float
        the gini index
    """
    gini_index = 1.0
    number_of_labels = len(labels)
    label_set = set(labels)
    for i in label_set:
        gini_index -= (1.0 * labels.count(i) / number_of_labels) ** 2
    return gini_index


def gain(feature, labels, feature_boundary):
    """
    Calculate the gain of a specific split. Use gini index as the measurement.
    And we do not need to compute the measurement of the parent node.

    Parameters
    -------------
    feature : list
        the feature values

    labels : list
        labels

    feature_boundary : float/int
        the split boundary of this feature

    Returns
    --------------
    gain : float
        the gain
    """
    number_of_labels = len(labels)
    labels_left = []
    labels_right = []
    for i, j in zip(feature, labels):
        if i <= feature_boundary:
            labels_left.append(j)
        else:
            labels_right.append(j)

    func = lambda label: -1.0 * len(label) / number_of_labels * gini(label)
    return func(labels_left) + func(labels_right)


def __build_tree(features, labels, root, features_pool, features_batch_size=None):
    """
    Build the decision tree

    Parameters
    -----------------
    features : 2-D array
        [n_sampels, m_features]

    labels : 1-D array
        binary labels for each sample

    root : Node
        the root of the new added tree (actually a branch to the existing tree)

    features_pool : set()
        from which features should be picked up one by one. In other words, the availables
        features to select from.

    features_batch_size : int, default=None
        The (maximum) number of features to be used in each node to calculate
        the split measurement during building decision tree.
        Leave it to be default (None) so as to use all (available) features.

    Returns
    ------------------
    A decision tree
    """
    # stop if all records in this node belong to the same class
    if len(set(labels)) == 1:
        root.label = labels[0]
        return root

    # stop if there is no records in this part
    if len(labels) < 1:
        return root

    # stop if no remaining features
    if not features_pool:
        root.label = majority_voting(labels)
        return root

    candidate_features = copy.deepcopy(features_pool) # we do NOT want features_pool to be modified
                                                      # so that deep copy should be used
    if features_batch_size and features_batch_size < len(features_pool):
        # if there are more features available, randomly pick up certain number
        candidate_features = set(random.sample(candidate_features, features_batch_size))

    # pick the feature from the pool to split the data
    # 1. for each feature, find the best split boundary
    # 2. pick the best split from all features
    split_candidate = []
    for feature_index in candidate_features:
        feature = [row[feature_index] for row in features]
        feature_values = set(feature) # all distinct feature values in a single column
        max_gain = -2.0
        boundary_temp = None
        for feature_boundary in feature_values:
            split_gain = gain(feature, labels, feature_boundary)
            if split_gain > max_gain:
                max_gain = split_gain
                boundary_temp = feature_boundary
        split_candidate.append((feature_index, boundary_temp, max_gain))

    max_split = max(split_candidate, key=lambda split: split[2])

    # split the data into two parts
    left_features = []
    left_labels = []
    right_features = []
    right_labels = []
    for row, label in enumerate(labels):
        if features[row][max_split[0]] <= max_split[1]:
            left_features.append(features[row])
            left_labels.append(label)
        else:
            right_features.append(features[row])
            right_labels.append(label)

    left_child = Node()
    right_child = Node()
    root.left_child = left_child
    root.right_child = right_child
    root.feature = max_split[0]
    root.feature_split_boundary = max_split[1]

    #__build_tree(left_features, left_labels, left_child, features_pool, features_batch_size)
    #__build_tree(right_features, right_labels, right_child, features_pool, features_batch_size)

    __build_tree(left_features, left_labels, left_child, features_pool - set([max_split[0]]), features_batch_size)
    __build_tree(right_features, right_labels, right_child, features_pool - set([max_split[0]]), features_batch_size)

    return root


def build_decision_tree(features, labels, features_batch_size=None):
    """
    Build the decision tree

    Parameters
    -----------------
    features : 2-D array
        [n_sampels, m_features]

    labels : 1-D array
        binary labels for each sample

    features_batch_size : set, default=None
        The (maximum) number of features to be used in each node to calculate
        the split measurement during building decision tree.
        Leave it to be default (None) so as to use all (available) features.

    Returns
    ------------------
    A decision tree
    """
    root = Node()
    features_pool = set(range(len(features[0])))
    return __build_tree(features, labels, root, features_pool, features_batch_size)


def classify_testing_record(tree, record):
    """
    Check the class label of one single testing record

    Parameters
    -----------------
    tree : Node()
        the 'root' (not necessary the real root) of the decision tree

    record : list
        the features of a single record

    Returns
    -----------------
    the class label in the leaf node
    """
    if not tree.left_child and not tree.right_child:
        return tree.label

    if record[tree.feature] <= tree.feature_split_boundary:
        return classify_testing_record(tree.left_child, record)
    else:
        return classify_testing_record(tree.right_child, record)


def classify_testing_dataset(d_tree, t_features):
    """
    Apply the testing data to the decision tree
    """
    result = []
    for r in t_features:
        result.append(classify_testing_record(d_tree, r))

    return result


if __name__ == "__main__":
    parameter_parser = argparse.ArgumentParser()
    parameter_parser.add_argument("-f", "--filename", default="project3_dataset2.txt", \
                                  help="file contains data")
    parameter_parser.add_argument("-k", "--num_of_folds", type=int, default=10, \
                                  help="number of fold-cross validation")
    args = parameter_parser.parse_args()

    features, labels, _ = help_functions.read_file(args.filename)
    # print(features[:, 3])

    overall_performance = []
    for i in range(args.num_of_folds):
        # split the data into training and testing
        training_features, testing_features = help_functions.get_traing_testing(features, args.num_of_folds, i)
        training_labels, testing_labels = help_functions.get_traing_testing(labels, args.num_of_folds, i)

        decision_tree = build_decision_tree(training_features, training_labels)

        classfied_labels = classify_testing_dataset(decision_tree, testing_features)
        performance = help_functions.compute_performance(testing_labels, classfied_labels)
        overall_performance.append(performance)
        print("fold " + str(i) + "'s performance: " + str(performance))

    overall_performance = np.array(overall_performance)
    print("\nOverall performance: " + str(overall_performance.mean(axis=0)))
