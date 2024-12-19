# import numpy as np
# import pandas as pd
# from linear import LR
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.metrics import mean_squared_error, r2_score


# #đọc dữ liệu
# file="\Python\Code_Python\Student_Performance.csv"
# data=pd.read_csv(file)
# #chuyển các ô không có giá trị thành giá trị 0
# # data = data.fillna(0)

# print(data)

# X = data.iloc[:-200, :-1].drop(data.columns[2],axis=1).values  # Tất cả các cột trừ cột cuối
# Y = data.iloc[:-200,-1].values.reshape(-1, 1)  #Lấy cột cuối

# print('X',X)
# print('Y',Y)

# #tạo ma trận 1
# one=np.ones((X.shape[0],1))

# # Chuẩn hóa X (dữ liệu đầu vào) bằng StandardScaler

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Cập nhật X với X_scaled đã chuẩn hóa
# Xbar = np.concatenate((np.ones((X_scaled.shape[0], 1)), X_scaled), axis=1)

# # w = np.zeros((Xbar.shape[1], 1))
# # m, n = Xbar.shape
# # w = np.zeros((n, 1))  # Khởi tạo weights với giá trị 0
# A = np.dot(Xbar.T, Xbar)
# b = np.dot(Xbar.T, Y)
# #m, n = Xbar.shape
# w = np.zeros((Xbar.shape[1], 1)) # Khởi tạo weights với giá trị 0
# #w=np.dot(np.linalg.pinv(np.dot(Xbar.T, Xbar)), np.dot(Xbar.T, Y))
# lr=LR(eta=0.001)
# # Tạo mô hình và chạy như bình thường
# w_new, i = lr.run(Xbar, Y, w)
# print('Trọng số w:',w_new)
# print('Vòng lặp kết thúc sau:',(i+1),'lần')

# #Lấy 200 dòng cuối trong dataset để kiểm tra dữ liệu
# X_thucte = data.iloc[-200: ,:].drop([data.columns[2], data.columns[-1]], axis=1).values
# Y_thucte = data.iloc[-200:,-1].values.reshape(-1, 1)  #Lấy cột cuối

# # Dự đoán với tập kiểm tra
# #Y_dudoan = np.dot(np.concatenate((np.ones((X_thucte.shape[0], 1)), X_thucte), axis=1), w_new)
# X_thucte_scaled = scaler.transform(X_thucte)

# # Thêm cột 1 vào X_thucte đã chuẩn hóa
# X_thucte_bar = np.concatenate((np.ones((X_thucte_scaled.shape[0], 1)), X_thucte_scaled), axis=1)

# # Dự đoán với tập kiểm tra
# Y_dudoan = np.dot(X_thucte_bar, w_new)
# # In kết quả
# print("Giá trị thực tế Y:",Y_thucte)
# print("Giá trị dự đoán Y:", Y_dudoan)

# r2 = r2_score(Y_thucte, Y_dudoan)
# print("R² score (độ chính xác): ",r2)


import numpy as np
import pandas as pd
from linear import LR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score


#đọc dữ liệu
file="\Python\Code_Python\Advertising.csv"
data=pd.read_csv(file)
#chuyển các ô không có giá trị thành giá trị 0
data = data.fillna(0)

print(data)

X = data.iloc[:-60, :-1].values  # Tất cả các cột trừ cột cuối
Y = data.iloc[:-60,-1].values.reshape(-1, 1)  #Lấy cột cuối

print('X',X)
print('Y',Y)

#tạo ma trận 1
one=np.ones((X.shape[0],1))

# Chuẩn hóa X (dữ liệu đầu vào) bằng StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cập nhật X với X_scaled đã chuẩn hóa
Xbar = np.concatenate((np.ones((X_scaled.shape[0], 1)), X_scaled), axis=1)

# w = np.zeros((Xbar.shape[1], 1))
# m, n = Xbar.shape
# w = np.zeros((n, 1))  # Khởi tạo weights với giá trị 0
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, Y)
#m, n = Xbar.shape
w = np.zeros((Xbar.shape[1], 1)) # Khởi tạo weights với giá trị 0
#w=np.dot(np.linalg.pinv(np.dot(Xbar.T, Xbar)), np.dot(Xbar.T, Y))
lr=LR(eta=0.001)
# Tạo mô hình và chạy như bình thường
w_new, i = lr.run(Xbar, Y, w)
print('Trọng số w:',w_new)
print('Vòng lặp kết thúc sau:',(i+1),'lần')

#Lấy 200 dòng cuối trong dataset để kiểm tra dữ liệu
X_thucte = data.iloc[-60: ,:-1].values
Y_thucte = data.iloc[-60:,-1].values.reshape(-1, 1)  #Lấy cột cuối

# Dự đoán với tập kiểm tra
#Y_dudoan = np.dot(np.concatenate((np.ones((X_thucte.shape[0], 1)), X_thucte), axis=1), w_new)
X_thucte_scaled = scaler.transform(X_thucte)

# Thêm cột 1 vào X_thucte đã chuẩn hóa
X_thucte_bar = np.concatenate((np.ones((X_thucte_scaled.shape[0], 1)), X_thucte_scaled), axis=1)

# Dự đoán với tập kiểm tra
Y_dudoan = np.dot(X_thucte_bar, w_new)
# In kết quả
print("Giá trị thực tế Y:")
print(Y_thucte)
print("Giá trị dự đoán Y:", Y_dudoan)

r2 = r2_score(Y_thucte, Y_dudoan)
print("R² score (độ chính xác): ",r2)
