# Tìm hiểu về Gradient Descent ứng dụng cho bài toán Linear Regression
## I. Giới thiệu chung.
### 1. Mô tả
Gradient Descent là một thuật toán tối ưu hóa được sử dụng rộng rãi trong học máy và thống kê để tìm cực tiểu của một hàm mất mát (loss function). Nó giúp điều chỉnh các tham số của mô hình sao cho sai số giữa giá trị dự đoán và giá trị thực tế nhỏ nhất.Dự án này nhằm tìm hiểu và triển khai thuật toán Gradient Descent trong bài toán hồi quy tuyến tính (Linear Regression). Dự án bao gồm lý thuyết cơ bản, cách áp dụng.
![Gradient-Descent-Top2](https://github.com/user-attachments/assets/e642c824-12fd-4b7c-82ae-e9f438783ee4)
### 2. Nội Dung Dự Án
Giới thiệu về Gradient Descent
Công thức toán học trong Linear Regression
Triển khai thuật toán bằng Python
Kết luận
## II. Giới thiệu về thuật toán Linear Regression.
### 1. Khái niệm
Linear Regression là một thuật toán học máy thuộc nhóm học có giám sát (Supervised Learning), được sử dụng để dự đoán các giá trị liên tục. Mô hình này thiết lập mối quan hệ tuyến tính giữa biến đầu vào (độc lập) và biến đầu ra (phụ thuộc).
### 2. Ý tưởng chính
Linear Regression giả định rằng có một mối quan hệ tuyến tính giữa biến đầu vào và biến đầu ra :

![CT1_Linear](https://github.com/user-attachments/assets/ffe339e8-0d24-438c-9217-caf776662995)

Dưới dạng vecto:

![Screenshot 2024-12-20 013845](https://github.com/user-attachments/assets/5af95bd8-e582-4857-af8f-587bbe7c220b)
### 3.Hàm mất mát

![Screenshot 2024-12-20 014004](https://github.com/user-attachments/assets/8e77b981-8feb-4759-b94e-66ec92c33f84)
