import numpy as np
import pandas as pd


class LinearModel():
    def __init__(self, alpha, reg_factor) -> None:
        self.reg_factor = reg_factor
        self.alpha = alpha
        self.costs = []

    @staticmethod
    def preprocess(X, y):
        X = np.array(X, ndmin=2)
        y = np.array(y, ndmin=2)

        if X.shape[1] == 1:
            X = X.T
        if y.shape[1] == 1:
            y = y.T

        return X, y
    
    def train(self, X, y, iters):
        X, y = self.preprocess(X, y)

        self.weights = np.zeros((1, X.shape[0]), dtype=np.float64)
        self.bias = np.random.rand()

        for i in range(iters):
            if i % 100 == 0:
                print(f"Completed {i+1} iterations")
            self._gradient_descent(X, y)


class LinearRegressor(LinearModel):
    def __init__(self, alpha, reg_factor) -> None:
        super().__init__(alpha, reg_factor)

    def _gradient_descent(self, X, y):
        prediction = self.weights.dot(X) + self.bias
        error = y - prediction
        m = X.shape[1]

        dw = (1 / m) * X.dot(error.T).T
        db = (1 / m) * (error).sum()

        self.weights = self.weights - self.alpha * dw
        self.bias = self.bias - self.alpha * db

        self.costs.append(self.__cost_function(m, error))
        

    def __cost_function(self, m, error):
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