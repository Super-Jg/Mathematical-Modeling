import numpy as np
import pandas as pd


# 判断级比
def lambda_ks(x0):
    lambda_k_arr = x0[:-1] / x0[1:]
    n = len(x0)
    Thate = [np.exp(-2 / (n + 1)), np.exp(2 / (n + 1))]
    if min(lambda_k_arr) > Thate[0] and max(lambda_k_arr) < Thate[1]:
        print("级比在可容覆盖内，可以使用GM(1,1)建模")
    else:
        print("级比不在可容覆盖内！请对序列作变换处理！！")

    return lambda_k_arr, Thate


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
data = np.arange(6).reshape(6,1)
x0 = data
for i in range(3):
    # 判断级比
    lambda_k, Thate = lambda_ks(x0)
    # 计算一次累加序列和
    x1 = sum_x1(x0)
    # 计算均值序列和
    z1 = aver_z1(x1)
    # 最小二乘法计算参数
    u = least_square_method(x0, z1)
    # 预测
    x0_pre = prediction(u, x1, 10 + i)
    # 误差分析
    delta_k, pho_k = error(x0, x0_pre, u, lambda_k)
    # 打印信息
    print("模型预测值为：")
    print(x0_pre[:len(x0)])
    print("相对误差为：")
    print(delta_k)
    print("级比误差为：")
    print(pho_k)
    print("2023年", int(7 + i), "月预测值为：")
    print(x0_pre[9 + i])
    x0 = np.append(x0, int(x0_pre[9 + i]))
print(x0)