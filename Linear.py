import numpy as np
# from GD import GD


class LR:
    def __init__(self,eta,Xbar,Y,w):
        self.eta=eta
        self.Xbar=Xbar
        self.Y=Y
        self.w=w

    def tinh_cost(self,Xbar,Y,w):
        predictions = np.dot(Xbar, w)
        return (1 / (2)) * np.sum((Y-predictions ) ** 2)
 