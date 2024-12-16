import numpy as np

class GD:
    def __init__(self,eta):
        self.eta=eta
        self.x_ds=[]
        

    def tinh_cost(self,X,Y,w):
        m = len(Y)
        predictions = np.dot(X,w)  
        cost = (1 / (2 * m)) * np.sum((Y-predictions) ** 2) 
        return cost


#f'(x) chung
    # def grad(self,x):
    #     eps=10**-3
    #     return (self.cost(x+eps)-self.cost(x-eps))/(2*eps)

    def tinh_grad(self,X,Y,w):
        m = len(Y)
        predictions = np.dot(X,w)  
        py = predictions - Y
        gradient = (1 / m) * np.dot(X.T,py )
        
        return gradient

#tính
    def run(self, X, Y,w):
        m = X.shape[0]

        w_cal = np.copy(w)

    # Vòng lặp tối đa 100 lần
        for i in range(10000):
            old_w_cal = np.copy(w_cal)
            gradient = self.tinh_grad(X, Y,w_cal)
            w_cal = old_w_cal - (self.eta * gradient)
            
            if np.linalg.norm(self.tinh_grad(X, Y,w_cal))/m < 1e-5 :  
                # print('Tìm ra giá trị sau',(i+1),'vòng lặp')

                break
        return w_cal,i


    

