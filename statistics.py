# -*- coding: utf-8 -*-

def readTrips():
      
    data_files = glob.glob('data2018/OD_2018-*.csv')
    
    li = []
    
    for filename in data_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    data = pd.concat(li, axis=0, ignore_index=True)
    data['start_date'] = pd.to_datetime(data['start_date'], format='%Y-%m-%d', errors='coerce')
    data['end_date'] = pd.to_datetime(data['end_date'], format='%Y-%m-%d', errors='coerce')
    
    return data

def readSTM():
    data_file = 'data2018/stops.txt'
    df = pd.read_csv(data_file)
    df = df[df['stop_id'].astype(str).str.startswith('STATION')]
    return df


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
    X_1 = dataDay[['Mean Temp (°C)']].values
    X_2 = dataDay[['Mean Temp (°C)']].values
    X_3 = dataDay[['Mean Temp (°C)']].values
    Y = dataDay[['departures_cnt']].values
    
    #Linear regression
    linear_regressor = LinearRegression()
    linear_regressor.fit(X, Y)
    Y_pred_linear = linear_regressor.predict(X)
    
    #polynmial degree 2 regression
    polynomial_feat = PolynomialFeatures(degree=2)
    x_poly_2 = polynomial_feat.fit_transform(X)
    polynomial_regressor = LinearRegression()
    polynomial_regressor.fit(x_poly_2, Y)
    
    Y_pred_poly_2 = polynomial_regressor.predict(x_poly_2)
    
    #polynmial degree 3 regression
    polynomial_feat_3 = PolynomialFeatures(degree=3)
    x_poly_3 = polynomial_feat_3.fit_transform(X)
    polynomial_regressor_3 = LinearRegression()
    polynomial_regressor_3.fit(x_poly_3, Y)
    
    Y_pred_poly_3 = polynomial_regressor_3.predict(x_poly_3)
    
    
    #Ploting the data
    plt.figure(figsize=(20, 10))
    plt.title('Scatter Plot Temp (°C) and Number of Trips')
    plt.scatter(X_1,Y,c='blue',marker='o')
    plt.xlabel('Mean Temp (°C)')
    plt.ylabel('Number of Daily Trips')
    plt.plot(X_1, Y_pred_linear, color='red')
    
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(X_2,Y_pred_poly_2), key=sort_axis)
    X_2, Y_pred_poly_2 = zip(*sorted_zip)
    plt.plot(X_2, Y_pred_poly_2, color='green')
    
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(X_3,Y_pred_poly_3), key=sort_axis)
    X_3, Y_pred_poly_3 = zip(*sorted_zip)
    plt.plot(X_3, Y_pred_poly_3, color='magenta')
    
    plt.plot(X_1, Y_pred_linear, '-r', label='degree=1')   
    plt.plot(X_2, Y_pred_poly_2, '-g', label='degree=2')   
    plt.plot(X_3, Y_pred_poly_3, '-m', label='degree=3')
    plt.legend(loc='upper left')
    
    plt.rcParams.update({'font.size': 22})
    plt.show()
    

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import glob
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import operator
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    
    #data = readTrips()
    #dataDay = summarizePerDay(data)
    #dataWeather = readWeather()
    #dataDay = mergeWeather(dataDay, dataWeather)
    
    dataSTM = readSTM()
    
    #scatterPlot(dataDay)
    
   


    
    
