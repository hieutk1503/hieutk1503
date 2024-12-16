import numpy as np
import pandas as pd
from linear import LR

#đọc dữ liệu
file="\Python\Student_Performance.csv"
data=pd.read_csv(file)
#chuyển các ô không có giá trị thành giá trị 0
# data = data.fillna(0)

print(data)

X = data.iloc[:, :-1].values  # Tất cả các cột trừ cột cuối
Y = data.iloc[:,-1].values.reshape(-1, 1)  #Lấy cột cuối

print('X',X)
print('Y',Y)

#tạo ma trận 1
one=np.ones((X.shape[0],1))
#thêm cột 1 vào ma trận X
Xbar=np.concatenate((one,X),axis=1)

# w = np.zeros((Xbar.shape[1], 1))
# m, n = Xbar.shape
# w = np.zeros((n, 1))  # Khởi tạo weights với giá trị 0
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, Y)
#m, n = Xbar.shape
#w = np.ones((n, 1))  # Khởi tạo weights với giá trị 0
print('Xbar:',Xbar,Xbar.shape)
w=np.dot(np.linalg.pinv(np.dot(Xbar.T, Xbar)), np.dot(Xbar.T, Y))
lr=LR(eta=0.01)
w_new,i = lr.run(Xbar,Y,w)

print('Trọng số w:',w_new)
print('Vòng lặp kết thúc sau:',(i+1),'lần')

w0=w_new[0,0]
y1 = (
    w_new[1, 0] * 2 + 
    w_new[2, 0] * 69 +
    w_new[3, 0] * 8 +
    w_new[4, 0] * 0 +
    w_new[0, 0]
)
print('y1=',y1)

