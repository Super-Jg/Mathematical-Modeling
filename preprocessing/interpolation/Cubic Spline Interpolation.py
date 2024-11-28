import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import datetime
#读取文件
df=pd.read_excel("附件1筛选.xlsx",engine='openpyxl')
df['月份'] = pd.to_datetime(df['月份'],format = '%Y-%m-%d %H:%M:%S')
print(df)
for row in df.index:
    for col in df.columns:
        if col == '月份' :
            df.loc[row,col] =df.at[row,'月份'].timestamp()
x=df.iloc[:,0].values
y=df.iloc[:,1].values
print(df)
'''if not np.all(np.isfinite(x)):
    print("x 数组中存在非有限值")
    x=x[np.isfinite(x)]
    y=y[np.isfinite(x)]
if not np.all(np.isfinite(y)):
    print("y 数组中存在非有限值")
    x=x[np.isfinite(y)]
    y=y[np.isfinite(y)]'''
x_interp=np.linspace(min(x),max(x),1000)
cs=CubicSpline(x,y)
y_interp=cs(x_interp)
plt.scatter(x,y,color='red',label='Original Data')
plt.plot(x_interp,y_interp,label='Cubic Spline')
plt.legend()
plt.show()
print(cs(1664553600))#2022年10月1日00:00:00时间戳