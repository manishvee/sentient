import numpy as np
import pandas as pd


class LinearModel():
    """
    Serves as a base class for all models that rely on optimising a linear function.

    Attributes:
    reg_factor : float
        Regularisation factor (lambda) to determine strnegth of regularisation applied on model.
    alpha : float
        Learning rate for gradient descent.
    costs : array
        Keeps track of computed loss after each iteration of gradient descent
    weights : ndarray
        Weights of each attribute in the linear model.
    bias : float
        Bias of the linear model.

    Methods:
    preprocess(X, y) :
        Transforms input data to ensure compatibility for matrix computations.
    train(X, y, iters) :
        Fits the linear model  to the given data.
    """
    def __init__(self, alpha, reg_factor) -> None:
        self.reg_factor = reg_factor
        self.alpha = alpha
        self.costs = []

    @staticmethod
    def __preprocess(X, y):
        """
        Transforms input data to ensure compatibility for matrix computations.

        Parameters:
        X : arraylike 
            Feature matrix on which model is trained
        y : arraylike
            Truth labels against which model is optimised.

        Returns:
            X and y after transformation
        """
        #convert X and y into numpy arrays 
        X = np.array(X, ndmin=2)
        y = np.array(y, ndmin=2)

        #ensure that dimensions of X and y are: (features, samples)
        if X.shape[1] == 1:
            X = X.T
        if y.shape[1] == 1:
            y = y.T

        return X, y
    
    def train(self, X, y, iters):
        """
        Fits the linear model  to the given data.

        Parameters:
        X : arraylike 
            Feature matrix on which model is trained
        y : arraylike
            Truth labels against which model is optimised.
        iters: int
            Number of iterations of gradient descent to be performed.
        """
        X, y = self.__preprocess(X, y)

        self.weights = np.zeros((1, X.shape[0]), dtype=np.float64)
        self.bias = np.random.rand()

        for i in range(1, iters+1):
            if i % 100 == 0:
                print(f"Completed {i} iterations")
            self._gradient_descent(X, y)


class LinearRegressor(LinearModel):
    """
    Implements regularised linear regression of the form y = w_1*x_1 + w_2*x_2 ... + b

    Attributes:
    reg_factor : float
        Regularisation factor (lambda) to determine strnegth of regularisation applied on model.
    alpha : float
        Learning rate for gradient descent.
    costs : array
        Keeps track of computed loss after each iteration of gradient descent
    weights : ndarray
        Weights of each attribute in the linear model.
    bias : float
        Bias of the linear model.

    Methods:
    gradient_descent(X, y) :
        Performs the gradient descent algorithm and updates the weights and bias of the model.
    cost_function(m, error) :
        Computes the root mean squared error of the model.
    """
    def __init__(self, alpha, reg_factor) -> None:
        super().__init__(alpha, reg_factor)

    def _gradient_descent(self, X, y):
        """
        Performs the gradient descent algorithm and updates the weights and bias of the model.

        Parameters:
        X : arraylike 
            Feature matrix on which model is trained
        y : arraylike
            Truth labels against which model is optimised.
        """
        prediction = self.weights.dot(X) + self.bias
        error = y - prediction
        m = X.shape[1]

        dw = (1 / m) * X.dot(error.T).T
        db = (1 / m) * (error).sum()

        self.weights = self.weights - self.alpha * dw
        self.bias = self.bias - self.alpha * db

        self.costs.append(self.__cost_function(m, error))
        
    def __cost_function(self, m, error):
        """
        Computes the root mean squared error of the model.

        Parameters:
        m : int
            Number of samples in the dataset
        error : ndarray
            The error (y_pred - y_real) for each sample in the dataset.

        Returns:
            Root mean squared error summed over all samples.
        """
        return (1 / (2 * m)) * (error ** 2).sum()


    def predict():
        pass

    def __str__(self) -> str:
        return f"Linear Regression Model \n{'*' * 25}\nWeights: {self.weights}\nBias: {self.bias}\nRMSE: {self.costs[-1]}"


class LogisticRegressor(LinearModel):
    def __init__(self) -> None:
        super().__init__()


def main():
    pass


if __name__ == "__main__":
    main()