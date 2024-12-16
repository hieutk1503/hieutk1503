import numpy as np
from GD import GD


class LR(GD):
    def __init__(self,eta):
        super().__init__(eta)

    def tinh_cost(self,Xbar,Y,w):
        m = len(Y)
        predictions = np.dot(Xbar, w)
        return (1 / (2 * m)) * np.sum((Y-predictions ) ** 2)
    
    def tinh_grad(self,Xbar,Y,w):
        m = len(Y)
        predictions = np.dot(Xbar, w)
        # print('pred2',predictions)

        return (1 / m) * np.dot(Xbar.T, (predictions -Y))

    


