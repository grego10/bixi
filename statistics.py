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

def readSTMStations():
    data_file = 'data2018/stops.txt'
    df = pd.read_csv(data_file)
    df = df[df['stop_id'].astype(str).str.startswith('STATION')]
    return df

dataSTM = readSTMStations()

def readBixiStations():
    data_file = 'data2018/Stations_2018.csv'
    df = pd.read_csv(data_file)
    return df

def bixiStationsDistance(bixiStations):
    #adding a column distance
    bixiStations['distance_metro'] = pd.Series( index=bixiStations.index)
    bixiStations['distance_metro'] = bixiStations['distance_metro'].astype('float64')
    
    #calculating distance
    bixiStations['distance_metro']= bixiStations.apply(calculateDistance, axis=1)
    
    return bixiStations

def mergeDist_TripCnt(dataBixi, dataTripCnt):
    
    dataTripCnt = dataTripCnt.rename(columns={'station_code':'code'})
    dataBixi = pd.merge(dataBixi, dataTripCnt, on='code')
    
    return dataBixi
    
def calculateDistance(row):
    latitude = row['latitude']
    longitude = row['longitude']
    
    smallest_distance = np.float64
    
    for index, row in dataSTM.iterrows():
        temp = calDist(latitude, longitude, row['stop_lat'], row['stop_lon'])
        if temp < smallest_distance:
            smallest_distance = temp
    
    return smallest_distance

def calDist(bixi_lat, bixi_lon, metro_lat, metro_lon):
    # The math module contains a function named radians which converts from degrees to radians. 
    bixi_lon = radians(bixi_lon) 
    metro_lon = radians(metro_lon) 
    bixi_lat = radians(bixi_lat) 
    metro_lat = radians(metro_lat) 
       
    # Haversine formula  
    dlon = metro_lon - bixi_lon  
    dlat = metro_lat - bixi_lat 
    a = sin(dlat / 2)**2 + cos(bixi_lat) * cos(metro_lat) * sin(dlon / 2)**2
  
    c = 2 * asin(sqrt(a))
     
    # Radius of earth in kilometers.
    r = 6371
       
    # calculate the result 
    return(c * r)

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

def stationsTripCnt(data):
    dataStart = data['start_station_code'].groupby(data['start_station_code']).count()
    dataStart = dataStart.to_frame('start_cnt').reset_index()
    dataStart = dataStart.rename(columns={'start_station_code':'station_code'})
    
    dataEnd = data['end_station_code'].groupby(data['end_station_code']).count()
    dataEnd = dataEnd.to_frame('end_cnt').reset_index()
    dataEnd = dataEnd.rename(columns={'end_station_code':'station_code'})
    
    data = pd.merge(dataStart, dataEnd, on='station_code')
    
    return data
    

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

def scatterPlotMetro(dataBixi):
    
    X = dataBixi[['distance_metro']].values
    X_1 = dataBixi[['distance_metro']].values
    X_2 = dataBixi[['distance_metro']].values
    Y_1 = dataBixi[['start_cnt']].values
    Y_2 = dataBixi[['end_cnt']].values       
    
    linear_regressor = LinearRegression()
    linear_regressor.fit(X_1, Y_1)
    Y_1_pred_linear = linear_regressor.predict(X_1)

    linear_regressor = LinearRegression()
    linear_regressor.fit(X_2, Y_2)
    Y_2_pred_linear = linear_regressor.predict(X_2)                
                  
    plt.figure(figsize=(20, 10))
    plt.title('Scatter Plot Distance Metro (km) and Number of Trips')
    plt.scatter(X_1,Y_1,c='blue',marker='o')
    plt.scatter(X_2,Y_2,c='red',marker='o')
    plt.xlabel('Distance from metro (km)')
    plt.ylabel('Number of Trips from a station in 2018')
    
    plt.plot(X_1, Y_1_pred_linear, color='blue')
    plt.plot(X_2, Y_2_pred_linear, color='red')
  
    plt.plot(X_1, Y_1, 'ob', label='start cnt')   
    plt.plot(X_2, Y_2, 'or', label='arrival cnt')   
    plt.legend(loc='upper right')
    
    plt.rcParams.update({'font.size': 18})
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
    from math import radians, cos, sin, asin, sqrt 
    
    data = readTrips()
    #dataDay = summarizePerDay(data)
    #dataWeather = readWeather()
    #dataDay = mergeWeather(dataDay, dataWeather)
    
    
    dataBixi = readBixiStations()
    
    dataBixi = bixiStationsDistance(dataBixi)
    dataTripCnt = stationsTripCnt(data)
    
    dataBixi = mergeDist_TripCnt(dataBixi, dataTripCnt)
    
    
    
    
    scatterPlotMetro(dataBixi)
    
    #scatterPlot(dataDay)


    
    
