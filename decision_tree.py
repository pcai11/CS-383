import csv
import math
import random


def read_data(csv_path):
    """Read in the training data from a csv file.
    
    The examples are returned as a list of Python dictionaries, with column names as keys.
    """
    examples = []
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for example in csv_reader:
            for k, v in example.items():
                if v == '':
                    example[k] = None
                else:
                    try:
                        example[k] = float(v)
                    except ValueError:
                        example[k] = v
            examples.append(example)
    return examples


def train_test_split(examples, test_perc):
    """Randomly data set (a list of examples) into a training and test set."""
    test_size = round(test_perc * len(examples))
    shuffled = random.sample(examples, len(examples))
    return shuffled[test_size:], shuffled[:test_size]


class TreeNodeInterface():
    """Simple "interface" to ensure both types of tree nodes must have a classify() method."""

    def classify(self, example):
        raise NotImplementedError


class DecisionNode(TreeNodeInterface):
    """Class representing an internal node of a decision tree."""

    def __init__(self, test_attr_name, test_attr_threshold, child_lt, child_ge, child_miss):
        """Constructor for the decision node.  Assumes attribute values are continuous.

        Args:
            test_attr_name: column name of the attribute being used to split data
            test_attr_threshold: value used for splitting
            child_lt: DecisionNode or LeafNode representing examples with test_attr_name
                values that are less than test_attr_threshold
            child_ge: DecisionNode or LeafNode representing examples with test_attr_name
                values that are greater than or equal to test_attr_threshold
            child_miss: DecisionNode or LeafNode representing examples that are missing a
                value for test_attr_name                 
        """
        self.test_attr_name = test_attr_name
        self.test_attr_threshold = test_attr_threshold
        self.child_ge = child_ge
        self.child_lt = child_lt
        self.child_miss = child_miss

    def classify(self, example):
        """Classify an example based on its test attribute value.
        
        Args:
            example: a dictionary { attr name -> value } representing a data instance

        Returns: a class label and probability as tuple
        """
        test_val = example[self.test_attr_name]
        if test_val is None:
            return self.child_miss.classify(example)
        elif test_val < self.test_attr_threshold:
            return self.child_lt.classify(example)
        else:
            return self.child_ge.classify(example)

    def __str__(self):
        return "test: {} < {:.4f}".format(self.test_attr_name, self.test_attr_threshold)


class LeafNode(TreeNodeInterface):
    """Class representing a leaf node of a decision tree.  Holds the predicted class."""

    def __init__(self, pred_class, pred_class_count, total_count):
        """Constructor for the leaf node.

        Args:
            pred_class: class label for the majority class that this leaf represents
            pred_class_count: number of training instances represented by this leaf node
            total_count: the total number of training instances used to build the leaf node
        """
        self.pred_class = pred_class
        self.pred_class_count = pred_class_count
        self.total_count = total_count
        self.prob = pred_class_count / total_count  # probability of having the class label

    def classify(self, example):
        """Classify an example.
        
        Args:
            example: a dictionary { attr name -> value } representing a data instance

        Returns: a class label and probability as tuple as stored in this leaf node.  This will be
            the same for all examples!
        """
        return self.pred_class, self.prob

    def __str__(self):
        return "leaf {} {}/{}={:.2f}".format(self.pred_class, self.pred_class_count,
                                             self.total_count, self.prob)


class DecisionTree:
    """Class representing a decision tree model."""

    def __init__(self, examples, id_name, class_name, min_leaf_count=1):
        """Constructor for the decision tree model.  Calls learn_tree().

        Args:
            examples: training data to use for tree learning, as a list of dictionaries
            id_name: the name of an identifier attribute (ignored by learn_tree() function)
            class_name: the name of the class label attribute (assumed categorical)
            min_leaf_count: the minimum number of training examples represented at a leaf node
        """
        self.id_name = id_name
        self.class_name = class_name
        self.min_leaf_count = min_leaf_count

        # build the tree!
        self.root = self.learn_tree(examples)

    def learn_tree(self, examples):
        """Build the decision tree based on entropy and information gain.
        
        Args:
            examples: training data to use for tree learning, as a list of dictionaries.  The
                attribute stored in self.id_name is ignored, and self.class_name is consided
                the class label.
        
        Returns: a DecisionNode or LeafNode representing the tree
        """
        classes = {}
        for ex in examples:
            if ex[self.class_name] not in classes:
                classes.update({ex[self.class_name]: 0})

        best_attr = ('attr', -math.inf, 0)
        left_subtree = []
        right_subtree = []
        best_misses = 0
        for attr in examples[0]:
            if attr == self.id_name:
                continue
            for i in range(len(examples)):
                threshold = examples[i][attr]
                if threshold is None:
                    continue
                lt = []
                ge = []
                misses = 0
                for ex in examples:
                    if ex[attr] is None:
                        misses += 1
                        continue
                    else:
                        if ex[attr] < threshold:
                            lt.append(ex)
                        elif ex[attr] >= threshold:
                            ge.append(ex)
                less_than = self.count_classes(lt, classes.copy())
                ge_than = self.count_classes(ge, classes.copy())

                length_lt = len(lt)
                length_ge = len(ge)

                if length_lt < self.min_leaf_count or length_ge < self.min_leaf_count:
                    continue

                if length_lt == 0 or length_ge == 0 or length_lt + length_ge - misses <= 0:
                    continue

                lt_probs = []
                ge_probs = []
                for item in classes:
                    lt_probs.append(less_than[item] / length_lt)
                    ge_probs.append(ge_than[item] / length_ge)

                has_zero = False
                probs = lt_probs + ge_probs
                for prob in probs:
                    if prob == 0:
                        has_zero = True
                if has_zero:
                    continue

                lt_entro = 0
                ge_entro = 0
                for item in lt_probs:
                    lt_entro += (item * math.log2(item))
                for item in ge_probs:
                    ge_entro += (item * math.log2(item))

                lt_prob = length_lt / (length_lt + length_ge - misses)
                info_gain = lt_prob * lt_entro + (1 - lt_prob) * ge_entro
                if info_gain > best_attr[1]:
                    best_misses = misses
                    best_attr = (attr, info_gain, threshold)
                    left_subtree = lt
                    right_subtree = ge

        if len(right_subtree) < self.min_leaf_count or len(left_subtree) < self.min_leaf_count:
            return self.get_leaf(self.count_classes(examples, classes.copy()), len(examples) - best_misses)
        else:
            left_tree = self.learn_tree(left_subtree)
            right_tree = self.learn_tree(right_subtree)
            if len(left_subtree) >= len(right_subtree):
                return DecisionNode(best_attr[0], best_attr[2], left_tree,
                                    right_tree, left_tree)
            else:
                return DecisionNode(best_attr[0], best_attr[2], left_tree,
                                    right_tree, right_tree)

    def count_classes(self, exs, classes):
        label = self.class_name
        for ex in exs:
            classes[ex[label]] += 1
        return classes

    def get_leaf(self, classes, total_size):
        max_class = ('attr', -1)

        for c in classes:
            if classes[c] > max_class[1]:
                max_class = (c, classes[c])

        return LeafNode(max_class[0], max_class[1], total_size)

    def classify(self, example):
        """Perform inference on a single example.

        Args:
            example: the instance being classified

        Returns: a tuple containing a class label and a probability
        """
        return self.root.classify(example)

    def __str__(self):
        """String representation of tree, calls _ascii_tree()."""
        ln_bef, ln, ln_aft = self._ascii_tree(self.root)
        return "\n".join(ln_bef + [ln] + ln_aft)

    def _ascii_tree(self, node):
        """Super high-tech tree-printing ascii-art madness."""
        indent = 6  # adjust this to decrease or increase width of output
        if type(node) == LeafNode:
            return [""], "leaf {} {}/{}={:.2f}".format(node.pred_class, node.pred_class_count, node.total_count,
                                                       node.prob), [""]
        else:
            child_ln_bef, child_ln, child_ln_aft = self._ascii_tree(node.child_ge)
            lines_before = [" " * indent * 2 + " " + " " * indent + line for line in child_ln_bef]
            lines_before.append(" " * indent * 2 + u'\u250c' + " >={}----".format(node.test_attr_threshold) + child_ln)
            lines_before.extend([" " * indent * 2 + "|" + " " * indent + line for line in child_ln_aft])

            line_mid = node.test_attr_name

            child_ln_bef, child_ln, child_ln_aft = self._ascii_tree(node.child_lt)
            lines_after = [" " * indent * 2 + "|" + " " * indent + line for line in child_ln_bef]
            lines_after.append(" " * indent * 2 + u'\u2514' + "- <{}----".format(node.test_attr_threshold) + child_ln)
            lines_after.extend([" " * indent * 2 + " " + " " * indent + line for line in child_ln_aft])

            return lines_before, line_mid, lines_after


def test_model(model, test_examples, label_ordering):
    """Test the tree on the test set and see how we did."""
    correct = 0
    almost = 0  # within one level of correct answer
    test_act_pred = {}
    for example in test_examples:
        actual = example[model.class_name]
        pred, prob = tree.classify(example)
        print("{:30} pred {:15} ({:.2f}), actual {:15} {}".format(example[id_attr_name] + ':',
                                                                  "'" + pred + "'", prob,
                                                                  "'" + actual + "'",
                                                                  '*' if pred == actual else ''))
        if pred == actual:
            correct += 1
        if abs(label_ordering.index(pred) - label_ordering.index(actual)) < 2:
            almost += 1
        test_act_pred[(actual, pred)] = test_act_pred.get((actual, pred), 0) + 1

    acc = correct / len(test_examples)
    near_acc = almost / len(test_examples)
    return acc, near_acc, test_act_pred


def confusion4x4(labels, vals):
    """Create an normalized predicted vs. actual confusion matrix for four classes."""
    n = sum([v for v in vals.values()])
    abbr = ["".join(w[0] for w in lab.split()) for lab in labels]
    s = ""
    s += " actual ___________________________________  \n"
    for ab, labp in zip(abbr, labels):
        row = [vals.get((labp, laba), 0) / n for laba in labels]
        s += "       |        |        |        |        | \n"
        s += "  {:^4s} | {:5.2f}  | {:5.2f}  | {:5.2f}  | {:5.2f}  | \n".format(ab, *row)
        s += "       |________|________|________|________| \n"
    s += "          {:^4s}     {:^4s}     {:^4s}     {:^4s} \n".format(*abbr)
    s += "                     predicted \n"
    return s


#############################################

if __name__ == '__main__':
    path_to_csv = 'town_vax_data.csv'
    id_attr_name = 'town'
    class_attr_name = 'vax_level'
    label_ordering = ['low', 'medium', 'high', 'very high']  # used to count "almost" right
    min_examples = 10  # minimum number of examples for a leaf node

    # read in the data
    examples = read_data(path_to_csv)
    train_examples, test_examples = train_test_split(examples, 0.25)

    # learn a tree from the training set
    tree = DecisionTree(train_examples, id_attr_name, class_attr_name, min_examples)

    # test the tree on the test set and see how we did
    acc, near_acc, test_act_pred = test_model(tree, test_examples, label_ordering)

    # print some stats
    print("\naccuracy: {:.2f}".format(acc))
    print("almost:   {:.2f}\n".format(near_acc))

    # visualize the results and tree in sweet, 8-bit text
    print(confusion4x4(label_ordering, test_act_pred))
    print(tree)

