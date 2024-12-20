import numpy as np
from Linear import LR
class GD(LR):
    def __init__(self,eta,Xbar,Y,w):
        super().__init__(eta,Xbar,Y,w)
        
#
    def tinh_cost(self,Xbar,Y,w):
        m = len(Y)# m= số dữ liệu
        predictions = np.dot(Xbar,w)  # giá trị của Y dự đoán
        cost = (1 / (2 * m)) * np.sum((Y-predictions) ** 2) # hàm mất mát
        return cost


#f'(x) chung
    # def grad(self,x):
    #     eps=10**-3
    #     return (self.cost(x+eps)-self.cost(x-eps))/(2*eps)

    def tinh_grad(self,Xbar,Y,w):
        m = len(Y) # m= chiều dài của Y
        predictions = np.dot(Xbar,w)  # giá trị của Y dựu đoán
        gradient = (1 / m) * np.dot(Xbar.T, (predictions -Y)) # đạo hàm hàm mất mát
        
        return gradient

#tính
    def run(self, Xbar, Y,w):
        m = Xbar.shape[0] #số dữu liệu

        w_cal = np.copy(w) # theta ban đầu

    # Vòng lặp tối đa 100 lần
        for i in range(100000):
            old_w_cal = np.copy(w_cal)
            gradient = self.tinh_grad(Xbar, Y,w_cal)
            w_cal = old_w_cal - (self.eta * gradient)
            if np.linalg.norm(self.tinh_grad(Xbar, Y,w_cal))/m < 1e-5 :  #thuật toán kết thúc khi giá trị đạo hàm nhỏ hơn 1e-5
                break
        return w_cal,i
    

