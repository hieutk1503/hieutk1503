import numpy as np
class GD:
    def __init__(self,eta,Xbar,Y,w):
        self.eta=eta
        self.Xbar=Xbar
        self.Y=Y
        self.w=w
        
#hàm mất mát
    def tinh_cost(self):
        m = len(self.Y)# m= số dữ liệu
        predictions = np.dot(self.Xbar,self.w)  # giá trị của Y dự đoán
        cost = (1 / (2 * m)) * np.sum((self.Y-predictions) ** 2) # hàm mất mát
        return cost
#đạo hàm hàm mất mát
    def tinh_grad(self):
        m = len(self.Y) # m= chiều dài của Y
        predictions = np.dot(self.Xbar,self.w)  # giá trị của Y dựu đoán
        gradient = (1 / m) * np.dot(self.Xbar.T, (predictions -self.Y)) # giá trị đạo hàm hàm mất mát
        return gradient

#tính
    def run(self,max_iter):
        m = self.Xbar.shape[0] #số dữu liệu
        for i in range(max_iter):# Vòng lặp 
            gradient = self.tinh_grad() #gái trị đạo hàm
            self.w = self.w - (self.eta * gradient)# công thức của gradient descent
            if np.linalg.norm(gradient)/m < 1e-5 :  #thuật toán kết thúc khi giá trị đạo hàm nhỏ hơn 1e-5
                break
        return self.w,i
    

