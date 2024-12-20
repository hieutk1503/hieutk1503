import numpy as np

class LR:
    def __init__(self,eta,X,Y,w):
        self.eta=eta
        self.X=X
        self.Y=Y
        self.w=w
#hàm mất mát
    def tinh_cost(self,X,Y,w):
        predictions = np.dot(X, w)
        return (1 / (2)) * np.sum((Y-predictions ) ** 2)
    


    


