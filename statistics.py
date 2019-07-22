# -*- coding: utf-8 -*-

def read():
      
    data_files = glob.glob('data2018/OD_2018-*.csv')
    
    li = []
    
    for filename in data_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    data = pd.concat(li, axis=0, ignore_index=True)
    data['start_date'] = pd.to_datetime(data['start_date'], format='%Y-%m-%d', errors='coerce')
    data['end_date'] = pd.to_datetime(data['end_date'], format='%Y-%m-%d', errors='coerce')
    
    return data



def summarizePerDay(data):
    data = ((data['start_date'].dt.date).groupby(data['start_date'].dt.date)).count()
    data = data.to_frame('departures_cnt').reset_index()
    data['start_date'] = pd.to_datetime(data['start_date'], format='%Y-%m-%d', errors='coerce')
    return data

def readWeather():
    data_file = 'data2018/WE_2018.csv'
    
    df = pd.read_csv(data_file)
    df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%m/%d/%Y', errors='coerce')
    
    df= df.rename(columns={'Date/Time':'start_date'})
    
    return df
    

def mergeWeather(dataDay, dataWeather):
    dataDay = pd.merge(dataDay, dataWeather[['start_date','Mean Temp (°C)']], on='start_date')
    return dataDay

def scatterPlot(dataDay):
    
    X = dataDay[['Mean Temp (°C)']].values
    Y = dataDay[['departures_cnt']].values
    
    linear_regressor = LinearRegression()
    linear_regressor.fit(X, Y)
    Y_pred_linear = linear_regressor.predict(X)
    
    polynomial_feat = PolynomialFeatures(degree=2)
    x_poly = polynomial_feat.fit_transform(X)
    polynomial_regressor = LinearRegression()
    polynomial_regressor.fit(x_poly, Y)
    
    Y_pred_poly = polynomial_regressor.predict(x_poly)
    
    plt.figure(figsize=(20, 10))
    plt.title('Scatter Plot Temp (°C) and Number of Trips')
    plt.scatter(X,Y,c='blue',marker='o')
    plt.xlabel('Mean Temp (°C)')
    plt.ylabel('Number of Daily Trips')
    plt.plot(X, Y_pred_linear, color='red')
    plt.plot(X, Y_pred_poly, color='green')
    
    plt.rcParams.update({'font.size': 22})
    plt.show()
    

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import glob
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    
    data = read()
    dataDay = summarizePerDay(data)
    dataWeather = readWeather()
    dataDay = mergeWeather(dataDay, dataWeather)
    
    scatterPlot(dataDay)
    
   


    
    
