import numpy as np
from GD import GD


class LR(GD):
    def __init__(self,eta):
        super().__init__(eta)
#     X = data.iloc[:,4 :-1].values  # Tất cả các cột trừ cột cuối cùng
#     Y = data.iloc[:,3].values.reshape(-1, 1)  # Cột cuối cùng

#     print('x=',X)
#     print('y=',Y)

# #tạo ma trận 1
#     one=np.ones((X.shape[0],1))
# #thêm cột 1 vào ma trận X
#     Xbar=np.concatenate((one,X),axis=1)

# #a=X^T*X
#     a=np.dot(Xbar.T,Xbar)
#     print('a=',a)
# #b=X^T*Y
#     b=np.dot(Xbar.T,Y)
#     print('b=',b)
# #w=(a^-1)*b
#     w=np.dot(np.linalg.pinv(a),b)
#     print('w=',w)
    def tinh_cost(self,Xbar,Y,w):
        m = len(Y)
        predictions = np.dot(Xbar, w)
        return (1 / (2 * m)) * np.sum((Y-predictions ) ** 2)
    
    def tinh_grad(self,Xbar,Y,w):
        m = len(Y)
        predictions = np.dot(Xbar, w)
        # print('pred2',predictions)

        return (1 / m) * np.dot(Xbar.T, (predictions -Y))

    


