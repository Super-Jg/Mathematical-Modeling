import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
#"df" refers to the dataframe obtained after importing an Excel file
df=pd.read_excel("./data.xlsx")
def cubic_spline_interpolation(df):
    #Retrieve columns A and B of the data and convert them into a Numpy array
    #Column A is x, and column B is y. Column A must be strictly increasing. If column A is not strictly increasing, the data must be sorted first. (If column A contains time-type data, you can convert the time-type data to timestamps before performing interpolation processing)
    x=df.iloc[:,0].values
    y=df.iloc[:,1].values
    #Set the target interpolation points, with "1000" as the number of interpolations
    x_interp=np.linspace(min(x),max(x),1000)
    #Perform cubic spline interpolation
    cs=CubicSpline(x,y)
    y_interp=cs(x_interp)
    #Plot the original data points
    plt.scatter(x,y,color='red',label='Original Data')
    #Plot the interpolation results
    plt.plot(x_interp,y_interp,label='Cubic Spline')
    plt.legend()
    plt.show()
