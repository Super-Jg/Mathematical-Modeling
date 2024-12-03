import numpy as np
import pandas as pd


# 判断级比
def lambda_ks(x0):
    lambda_k_arr = x0[:-1] / x0[1:]
    n = len(x0)
    Thate = [np.exp(-2 / (n + 1)), np.exp(2 / (n + 1))]
    flag = True# 判断是否需要对序列进行平移变换
    if min(lambda_k_arr) > Thate[0] and max(lambda_k_arr) < Thate[1]:
        print("级比在可容覆盖内，可以使用GM(1,1)建模")
    else:
        print("级比不在可容覆盖内！请对序列作变换处理！！")
        flag = False

    return lambda_k_arr, Thate , flag


# 产生累加序列
def sum_x1(x0):
    return np.cumsum(x0)


# 产生均值生成序列
def aver_z1(x1):
    arr1 = x1[:-1]
    arr2 = x1[1:]
    z1 = (arr1 + arr2) / 2
    return z1


# 最小二乘法
def least_square_method(x0, z1):
    Y = np.zeros_like(x0[1:])
    Y[:] = x0[1:]
    B = np.zeros((len(z1), 2))
    B[:, 0] = z1[:] * (-1)
    B[:, 1] = 1

    u = np.linalg.inv(B.T @ B) @ B.T @ Y

    return u


# 模型预测
def prediction(u, x1, n):
    x1_k_add_1_ls = []
    x1_k_add_1_ls.append(x1[0])
    for i in range(1, n):
        x1_kadd1 = (x1[0] - u[1] / u[0]) * np.exp(-u[0] * i) + u[1] / u[0]
        x1_k_add_1_ls.append(x1_kadd1)
    x1_k_add_1_arr = np.array(x1_k_add_1_ls)
    x0_pre = np.zeros(n)
    x0_pre[0] = x1[0]
    arr1 = x1_k_add_1_arr[1:] - x1_k_add_1_arr[:-1]
    x0_pre[1:] = arr1[:]

    return x0_pre


# 误差计算
def error(x0, x0_pre, u, lambda_k):
    delta_k = np.abs(x0 - x0_pre[:len(x0)]) / x0
    pho_k = np.abs(1 - (1 - 0.5 * u[0]) / (1 + 0.5 * u[0]) * lambda_k)

    return delta_k, pho_k

# 导入数据
data = pd.read_excel("./data.xlsx")
# x0为用于预测的历史数据
x0 = np.array(data.iloc[:, 1])
def grey_prediction_model(x0,n):
    # 需要根据x0历史数据预测接下来的n年或n月或n日等的数据
    for i in range(n):
        sum = 0
        # 判断级比
        lambda_k, Thate , flag = lambda_ks(x0)
        # 判断是否进行平移变换
        while flag == False:
            x0 = x0 + 100
            sum = sum + 1
            lambda_k, Thate, flag = lambda_ks(x0)
        # 计算一次累加序列和
        x1 = sum_x1(x0)
        # 计算均值序列和
        z1 = aver_z1(x1)
        # 最小二乘法计算参数
        u = least_square_method(x0, z1)
        # 预测
        x0_pre = prediction(u, x1, len(x0) + 1 + i)
        # 误差分析
        delta_k, pho_k = error(x0, x0_pre, u, lambda_k)
        # 打印信息
        print("模型预测值为：")
        print(x0_pre[:len(x0)]-sum*100)
        print("相对误差为：")
        print(delta_k)
        print("级比误差为：")
        print(pho_k)
        x0 = np.append(x0, int(x0_pre[len(x0) + i]))-sum*100
    print(x0)
