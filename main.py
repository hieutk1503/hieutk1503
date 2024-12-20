import numpy as np
import pandas as pd
from GD import GD
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

#đọc dữ liệu
def doc_du_lieu():
    file = "\\Python\\Code_Python\\Student_Performance.csv"
    data=pd.read_csv(file)
    return data

def xu_li_du_lieu(data):
    X = data.iloc[:-200, :-1].drop(data.columns[2],axis=1).values  # Tất cả các cột trừ cột cuối và bỏ cột 3
    Y = data.iloc[:-200,-1].values.reshape(-1, 1)  #Lấy cột cuối
    return X,Y

def cap_nhat_Xbar(X):
    one=np.ones((X.shape[0],1))#tạo ma trận 1
    scaler = StandardScaler()# Chuẩn hóa X (dữ liệu đầu vào) bằng StandardScaler
    X_scaled =scaler.fit_transform(X)

    Xbar = np.concatenate((one, X_scaled), axis=1)# Cập nhật X với X_scaled đã chuẩn hóa
    return Xbar
if __name__=="__main__":
    data=doc_du_lieu()
    X,Y=xu_li_du_lieu(data)
    Xbar=cap_nhat_Xbar(X)
    w = np.zeros((Xbar.shape[1], 1)) # Khởi tạo w với giá trị 0
    print(Xbar,Y)
    gd=GD(eta=0.001,Xbar=Xbar,Y=Y,w=w)

    w_new, i = gd.run(Xbar, Y, w)

    print('Trọng số w:',w_new)
    print('Vòng lặp kết thúc sau:',(i+1),'lần')

#Lấy 200 dòng cuối trong dataset để kiểm tra dữ liệu
    X_thucte = data.iloc[-200: ,:].drop([data.columns[2], data.columns[-1]], axis=1).values
    Y_thucte = data.iloc[-200:,-1].values.reshape(-1, 1)  #Lấy cột cuối

# Dự đoán với tập kiểm tra
    X_thucte_bar = cap_nhat_Xbar(X_thucte)
# Dự đoán với tập kiểm tra
    Y_dudoan = np.dot(X_thucte_bar, w_new)
# In kết quả
    print("Giá trị thực tế Y:",Y_thucte.T)
    print("Giá trị dự đoán Y:", Y_dudoan.T)
    r2 = r2_score(Y_thucte, Y_dudoan)
    print("Độ chính xác: ",r2)

