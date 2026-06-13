import numpy as np
from util import sigmoid, euclidean_distance
from metrics import accuracy, Confusion_Matrix, precision, recall, F1_score
from collections import Counter
import pandas as pd
from collections import defaultdict
import math

class BaseModel:
    def fit(self, X, y):
        raise NotImplementedError
    def predict(self, X):
        raise NotImplementedError
    def evaluate(self, X, y):
        raise NotImplementedError

class LogisticRegression(BaseModel):
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.losses = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_01 = np.where(y == 1, 1, 0)
        
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.losses = []
        
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear_model)
            
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y_01))
            db = (1 / n_samples) * np.sum(y_pred - y_01)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            loss = -np.mean(y_01 * np.log(y_pred + 1e-15) + (1 - y_01) * np.log(1 - y_pred + 1e-15))
            self.losses.append(loss)

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_model)
        
        return np.where(y_pred >= 0.5, 1, -1)

    def evaluate(self, X, y):
       
        
        y_pred = self.predict(X)
        return {
            'model': 'Logistic Regression',
            'accuracy': accuracy(y, y_pred),
            'confusion_matrix': Confusion_Matrix(y, y_pred),
            'precision': precision(y, y_pred),
            'recall': recall(y, y_pred),
            'f1_score': F1_score(y, y_pred)
        }

class KNN(BaseModel):
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def evaluate(self, X, y):
  
        y_pred = self.predict(X)
        return {
            'model': 'KNN',
            'accuracy': accuracy(y, y_pred),
            'confusion_matrix': Confusion_Matrix(y, y_pred),
            'precision': precision(y, y_pred),
            'recall': recall(y, y_pred),
            'f1_score': F1_score(y, y_pred)
        }

class LinearSVM(BaseModel):
    def __init__(self, learning_rate=0.0001, lambda_param=0.001, n_iters=5000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.n_iters):
            approx = np.dot(X, self.w) - self.b
            condition = y * approx >= 1
            
            dw = 2 * self.lambda_param * self.w
            violating_mask = ~condition
            
            if np.any(violating_mask):
                dw -= np.dot(X[violating_mask].T, y[violating_mask]) / n_samples
                db = -np.sum(y[violating_mask]) / n_samples
            else:
                db = 0.0

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

    def evaluate(self, X, y):
        # Clean: No redundant 'np.where' re-mappings needed!
        y_true = y.ravel()
        y_pred = self.predict(X)
        return {
            'model': 'Linear SVM',
            'accuracy': accuracy(y_true, y_pred),
            'confusion_matrix': Confusion_Matrix(y_true, y_pred),
            'precision': precision(y_true, y_pred),
            'recall': recall(y_true, y_pred),
            'f1_score': F1_score(y_true, y_pred)
        }
    
class KernelPerceptron(BaseModel):

    def __init__(
        self,
        kernel='rbf',
        gamma=0.01,
        degree=2,
        coef0=1,
        epochs=50
    ):

        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.epochs = epochs

    def _kernel(self, X1, X2):

        if self.kernel == 'rbf':

            sq1 = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            sq2 = np.sum(X2 ** 2, axis=1)

            dists = sq1 + sq2 - 2 * np.dot(X1, X2.T)

            return np.exp(
                -self.gamma * dists
            )

        elif self.kernel == 'poly':

            return (
                np.dot(X1, X2.T) / X1.shape[1]
                + self.coef0
            ) ** self.degree

        else:
            raise ValueError(
                f"Unknown kernel '{self.kernel}'"
            )

    def fit(self, X, y):

        self.X_train = X
        self.y_train = y.ravel()

        n_samples = X.shape[0]

        self.alpha = np.zeros(n_samples)
      

        K = self._kernel(X, X)

        
        self.training_errors = []

        for epoch in range(self.epochs):

            errors = 0

            for i in range(n_samples):

                decision = (
                    np.sum(
                        self.alpha
                        * self.y_train
                        * K[:, i]
                    )
                    
                )

                prediction = (
                    1 if decision >= 0
                    else -1
                )

                if prediction != self.y_train[i]:

                    self.alpha[i] += 2


                    errors += 1

            self.training_errors.append(errors)

            if errors == 0:
                break

        self.support_vectors_ = np.sum(
            self.alpha > 0
        )

        print(
            f"Training finished after {epoch+1} epochs"
        )

        print(
            f"Support vectors: {self.support_vectors_}"
        )

        return self

    def predict(self, X):

        K_pred = self._kernel(
            X,
            self.X_train
        )

        decision = (
            np.dot(
                K_pred,
                self.alpha * self.y_train
            )
            
        )

        return np.where(
            decision >= 0,
            1,
            -1
        )

    def evaluate(self, X, y):

        y_true = y.ravel()
        y_pred = self.predict(X)

        return {
            'model': f'Kernel Perceptron ({self.kernel.upper()})',
            'accuracy': accuracy(y_true, y_pred),
            'confusion_matrix': Confusion_Matrix(y_true, y_pred),
            'precision': precision(y_true, y_pred),
            'recall': recall(y_true, y_pred),
            'f1_score': F1_score(y_true, y_pred)
        }
    


class BinaryCrossEntropyLoss:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def loss(labels, predictions):
        probs = BinaryCrossEntropyLoss.sigmoid(predictions)
        
        # To avoid log(0)
        epsilon = 1e-15
        
        probs = np.clip(probs, epsilon, 1 - epsilon)
        
        # Binary log loss
        return -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
    @staticmethod
    def gradients(labels, predictions):
        probs = BinaryCrossEntropyLoss.sigmoid(predictions)
        
        # Gradient of binary cross-entropy
        return probs - labels

    @staticmethod
    def hessians(labels, predictions):
        probs = BinaryCrossEntropyLoss.sigmoid(predictions)
        
        # Hessian for sigmoid cross-entropy
        return probs * (1 - probs)
    

class BoostedTree:
    """
    A single decision tree within a gradient boosting ensemble.
    
    This class implements a regression tree that optimizes splits based on 
    gradients and hessians from the loss function. It follows the XGBoost algorithm
    principles for building trees in a boosted ensemble.
    
    Parameters
    ----------
    X : numpy.ndarray or pandas.DataFrame
        The feature matrix for training
    gradients : numpy.ndarray or pandas.Series
        First-order gradients of the loss function
    hessians : numpy.ndarray or pandas.Series
        Second-order gradients (hessians) of the loss function
    params : dict
        Dictionary of hyperparameters:
        - 'min_child_weight': Minimum sum of hessian needed in a child node
        - 'reg_lambda': L2 regularization term
        - 'gamma': Minimum loss reduction to make a split
    max_depth : int
        Maximum depth of the tree
    idxs : numpy.ndarray, optional
        Indices of the samples to use for building this tree
    """
    def __init__(self, X, gradients, hessians, params, max_depth, idxs=None):
        self.X = X.values if isinstance(X, pd.DataFrame) else X
        self.gradients = gradients.values if isinstance(gradients, pd.Series) else gradients
        self.hessians = hessians.values if isinstance(hessians, pd.Series) else hessians
        self.params = params
        self.min_child_weight = self.params['min_child_weight'] if self.params['min_child_weight'] else 1.0
        self._lambda = self.params['reg_lambda'] if self.params['reg_lambda'] else 1.0
        self.gamma = self.params['gamma'] if self.params['gamma'] else 0.0
        self.max_depth = max_depth
        self.ridxs = idxs if idxs is not None else np.arange(len(gradients))
        self.num_examples = len(self.ridxs)  # Number of training examples
        self.num_features = X.shape[1]  # Number of features
        self.weight = -self.gradients[self.ridxs].sum() / (self.hessians[self.ridxs].sum() + self._lambda)  # Leaf weight
        self.split_score = 0.0  # Best gain so far
        self.split_idx = 0  # Feature index for split
        self.threshold = 0.0  # Threshold value for split
        self._build_tree_structure()  # Recursively build the tree
    def _build_tree_structure(self):
        """
        Recursively builds the tree structure by finding the best splits.
        
        This method attempts to find the best split for the current node by evaluating
        all possible features. If a valid split is found, it creates left and right
        child nodes recursively until max_depth is reached or no valid split is found.
        
        Returns
        -------
        None
        """
        if self.max_depth <= 0:
            return  # Reached max depth, stop recursion

        for fidx in range(self.num_features):
            self._find_best_split_score(fidx)  # Try splitting on each feature
        if self._is_leaf:
            return  # No valid split found, stop here

        feature = self.X[self.ridxs, self.split_idx]
        left_idxs = np.nonzero(feature <= self.threshold)[0]
        right_idxs = np.nonzero(feature > self.threshold)[0]

        # Recursively build left and right subtrees
        self.left = BoostedTree(self.X, self.gradients, self.hessians, self.params,
                                self.max_depth - 1, self.ridxs[left_idxs])
        self.right = BoostedTree(self.X, self.gradients, self.hessians, self.params,
                                self.max_depth - 1, self.ridxs[right_idxs])
    def _find_best_split_score(self, fidx):
        """
        Finds the best splitting point for a given feature.
        
        This method evaluates all possible split points for the given feature and
        computes the gain for each. It updates the tree's split information if a 
        better split than the current best is found.
        
        Parameters
        ----------
        fidx : int
            Index of the feature to evaluate for splitting
            
        Returns
        -------
        None
        """
        feature = self.X[self.ridxs, fidx]
        gradients = self.gradients[self.ridxs]
        hessians = self.hessians[self.ridxs]

        sorted_idxs = np.argsort(feature)
        sorted_feature = feature[sorted_idxs]
        sorted_gradient = gradients[sorted_idxs]
        sorted_hessians = hessians[sorted_idxs]

        hessian_sum = sorted_hessians.sum()
        gradient_sum = sorted_gradient.sum()

        right_hessian_sum = hessian_sum
        right_gradient_sum = gradient_sum
        left_hessian_sum = 0.0
        left_gradient_sum = 0.0

        for idx in range(0, self.num_examples - 1):
            candidate = sorted_feature[idx]
            neighbor = sorted_feature[idx + 1]
            gradient = sorted_gradient[idx]
            hessian = sorted_hessians[idx]

            right_gradient_sum -= gradient
            right_hessian_sum -= hessian
            left_gradient_sum += gradient
            left_hessian_sum += hessian

            if right_hessian_sum < self.min_child_weight:
                continue

            if left_hessian_sum < self.min_child_weight:
                continue
                # Compute gain from potential split
            right_score = (right_gradient_sum ** 2) / (right_hessian_sum + self._lambda)
            left_score = (left_gradient_sum ** 2) / (left_hessian_sum + self._lambda)
            score_before_split = (gradient_sum ** 2) / (hessian_sum + self._lambda)
            gain = 0.5 * (left_score + right_score - score_before_split) - self.gamma
            if gain > self.split_score:
                self.split_score = gain
                self.split_idx = fidx
                self.threshold = (candidate + neighbor) / 2
                    
    def _predict_row(self, example):
        """
        Make a prediction for a single example by traversing the tree.
        
        This method traverses the tree from root to leaf based on the feature
        values of the given example, and returns the weight of the leaf node.
        
        Parameters
        ----------
        example : numpy.ndarray
            A single example as an array of feature values

            Returns
        -------
        float
            The prediction value for this example
        """
        if self._is_leaf:
            return self.weight  # Return leaf weight
        child = self.left if example[self.split_idx] <= self.threshold else self.right
        return child._predict_row(example)  # Recurse down the tree
    def predict(self, X):

        X = X.values if isinstance(X, pd.DataFrame) else X

        return np.array([
            self._predict_row(row)
            for row in X
        ])

    @property
    def _is_leaf(self):
        """
        Determines if the current node is a leaf node.
        
        Returns
        -------
        bool
            True if this is a leaf node (no valid split found), False otherwise
        """
        return self.split_score == 0.0  # Leaf node if no gain found
class XGBoost:
    """
    XGBoost implementation for gradient boosting.
    
    This class implements a gradient boosting algorithm inspired by XGBoost.
    It builds an ensemble of BoostedTree models to make predictions by
    sequentially adding trees that correct the errors of previous ones.
    
    Parameters
    ----------
    params : dict
        Dictionary of hyperparameters:
        - 'subsample': Fraction of training examples to use for each tree
        - 'base_score': Initial prediction value for all instances
        - 'learning_rate': Step size shrinkage used to prevent overfitting
        - 'max_depth': Maximum depth of each tree
        - Other parameters passed to BoostedTree (min_child_weight, reg_lambda, gamma)
    objective : object
        Objective function that implements gradients() and hessians() methods
    seed : int, default=42
        Random seed for reproducibility
    """
    def __init__(self, params, objective, seed=42):
        self.trees = []  # Store all trained trees
        self.params = defaultdict(lambda: None, params)  # Default values for missing params
        self.objective = objective  # Loss function
        self.subsample = self.params['subsample'] if self.params['subsample'] else 1.0
        self.base_score = self.params['base_score'] if self.params['base_score'] else 0.5
        self.learning_rate = self.params['learning_rate'] if self.params['learning_rate'] else 1e-1
        self.max_depth = self.params['max_depth'] if self.params['max_depth'] else 5
        self.rng = np.random.default_rng(seed=seed)  # Random number generator
        
    def fit(self, X, y, num_rounds):
        """
        Fit the XGBoost model to training data.
        
        Trains an ensemble of trees sequentially, where each tree attempts to
        correct the errors made by the previous trees. The process involves:
        1. Computing gradients and hessians based on current predictions
        2. Building a new tree to minimize these gradients
        3. Adding the tree's predictions (scaled by learning rate) to the ensemble
        
        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Training features
        y : numpy.ndarray or pandas.Series
            Target values
        num_rounds : int
            Number of boosting rounds (trees) to build
            
        Returns
        -------
        None
        """
        if self.base_score == 0.5:

            p = np.mean(y)

            p = np.clip(
                p,
                1e-6,
                1 - 1e-6
            )

            self.base_score = np.log(
                p / (1 - p)
            )
        predictions = self.base_score * np.ones(shape=y.shape)  # Initialize predictions
        for rnd in range(num_rounds):
            gradients = self.objective.gradients(y, predictions)  # Compute gradients
            hessians = self.objective.hessians(y, predictions)  # Compute hessians
            # Row sampling
            idxs = None if self.subsample == 1.0 else self.rng.choice(
                len(y),
                size=math.floor(self.subsample * len(y)),
                replace=False
            )
            # Train one tree on the current gradients
            tree = BoostedTree(
                X=X,
                gradients=gradients,
                hessians=hessians,
                params=self.params,
                max_depth=self.max_depth,
                idxs=idxs
            )
            self.trees.append(tree)
            predictions += self.learning_rate * tree.predict(X)  # Update predictions

    def predict(self, X):
        """
        Make predictions using the trained XGBoost model.
        
        Computes predictions by starting with the base score and adding
        the weighted contributions from all trees in the ensemble.
        
        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Features to make predictions on
            
        Returns
        -------
        numpy.ndarray
            Predicted values for each input example
        """
        # Add predictions from all trees
        return self.base_score + self.learning_rate * np.sum([tree.predict(X) for tree in self.trees], axis=0)
class XGBoostSigmoid:
    """
    XGBoost classifier for binary classification problems using sigmoid activation.
    
    This class provides a wrapper around the XGBoost class specifically for
    binary classification. It uses a sigmoid function to convert raw model outputs
    into probability scores, which can then be thresholded to obtain binary predictions.
    
    Parameters
    ----------
    params : dict
        Dictionary of hyperparameters to be passed to the underlying XGBoost model.
        See XGBoost class documentation for details on supported parameters.
    threshold : float, default=0.5
        Decision threshold for binary classification. Probability scores above
        this threshold are classified as 1, otherwise 0.
    seed : int, default=42
        Random seed for reproducibility.
    """
    def __init__(self, params, threshold=0.5, seed=42):
        self.params = params
        self.threshold = threshold  # Threshold to classify sigmoid output as 0 or 1
        self.objective = BinaryCrossEntropyLoss()
        self.base = XGBoost(self.params, self.objective, seed)
    def fit(self, X, y):
        """
        Unified interface used by run_experiments().
        Converts labels from {-1,1} to {0,1} before training.
        """
        y_01 = np.where(y == -1, 0, 1)

        self.train(
            X,
            y_01,
            num_rounds=50
        )
    def train(self, X, y, num_rounds):
        """
        Train the XGBoostSigmoid classifier.
        
        Trains the underlying XGBoost model using binary cross-entropy loss.
        
        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Training features.
        y : numpy.ndarray or pandas.Series
            Binary target values (0 or 1).
        num_rounds : int
            Number of boosting rounds (trees) to build.
            
        Returns
        -------
        None
        """
        self.base.fit(X, y, num_rounds)  # Train the underlying boosted trees
    def predict(self, X, with_labels=False, threshold=0.5):
        """
        Make predictions using the trained XGBoostSigmoid model.
        
        Computes raw scores using the underlying XGBoost model, then applies
        a sigmoid function to obtain probability scores. Optionally returns
        binary class labels based on a threshold.
        
        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Features to make predictions on.
        with_labels : bool, default=False
            If True, returns both probability scores and binary class labels.
            If False, returns only probability scores.
        threshold : float, default=0.5
            Decision threshold for binary classification. Overrides the threshold
            set during initialization if provided.
            
        Returns
        -------
        numpy.ndarray or tuple of numpy.ndarray
            If with_labels=False, returns probability scores.
            If with_labels=True, returns a tuple of (probability_scores, binary_labels).
        """
        logits = self.base.predict(X)  # Get raw scores
        probs = self.objective.sigmoid(logits)  # Apply sigmoid to get probabilities
        if with_labels:
            return probs, (probs >= threshold).astype(int)
        return probs
    def evaluate(self, X, y):

        y_true = y.ravel()

        probs, y_pred = self.predict(
            X,
            with_labels=True
        )
        y_pred = np.where(y_pred == 0, -1, 1)

        return {
            'model': 'XGBoost',
            'accuracy': accuracy(y_true, y_pred),
            'confusion_matrix': Confusion_Matrix(y_true, y_pred),
            'precision': precision(y_true, y_pred),
            'recall': recall(y_true, y_pred),
            'f1_score': F1_score(y_true, y_pred)
        }