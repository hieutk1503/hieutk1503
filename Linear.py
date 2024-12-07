# LinearRegression.py
import numpy as np
from GD import GD  # Import lớp GD từ file GD.py

class LinearRegression(GD):
    def __init__(self, learning_rate=0.01, max_iter=1000, tolerance=1e-6):
        super().__init__(learning_rate, max_iter, tolerance)
    
    def compute_cost(self, X, Y):
        """
        Hàm chi phí: Mean Squared Error (MSE).
        :param X: Ma trận đặc trưng
        :param Y: Nhãn (output)
        """
        m = len(Y)
        predictions = np.dot(X, self.weights)
        return (1 / (2 * m)) * np.sum((predictions - Y) ** 2)
    
    def compute_gradient(self, X, Y):
        """
        Tính gradient của hàm chi phí.
        :param X: Ma trận đặc trưng
        :param Y: Nhãn (output)
        """
        m = len(Y)
        predictions = np.dot(X, self.weights)
        return (1 / m) * np.dot(X.T, (predictions - Y))