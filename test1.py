# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 14:44:46 2019

@author: greg0
"""
# Load journey data

def read():
    f = 'OD_2019-05.csv'
    data = pd.read_csv(f)

    data['start_date'] = pd.to_datetime(data['start_date'], format='%Y-%m-%d', errors='coerce')
    data['day_number'] = pd.to_datetime(data['start_date']).dt.dayofweek
    data['week_day_b'] = pd.Series( index=data.index)
    data['week_day_b'] = 0
    data.loc[data['day_number'].isin([5, 6]), 'week_day_b'] = 1
    data['time_slice'] = data['start_date'].dt.hour * 60 + np.floor(data['start_date'].dt.minute/20)*20
    
    return data

def visualizePerDay(data, column_name, color='#494949', title='Departure per day'):
  
    plt.figure(figsize=(20, 10))
    ax = (data[column_name].groupby(data[column_name].dt.day)
                         .count()).plot(kind="bar", color=color)
    ax.set_facecolor('#eeeeee')
    ax.set_xlabel("time of day")
    ax.set_ylabel("number of trips")
    ax.set_title(title)
    plt.show()
    
def visualizePerHourW(data, column_name, color='#494949', title='Departure per hour (Week)'):
    
    #WEEK
    dataWeek = data.drop(data[data['week_day_b'] == 1].index)
    plt.figure(figsize=(20, 10))
    ax = (dataWeek[column_name].groupby(dataWeek[column_name])
                         .count()).plot(kind="bar", color=color)
    ax.set_facecolor('#eeeeee')
    ax.set_xlabel("Hour of day")
    ax.set_xticks(np.arange(0, 75, 3))
    ax.set_xticklabels(np.arange(0, 24, 1)) 
    ax.set_ylabel("Number of trips")
    ax.set_title(title)
    plt.show()

def visualizePerHourE(data, column_name, color='#494949', title='Departure per hour (Weekend)'):
    
    #WEEK
    dataWeek = data.drop(data[data['week_day_b'] == 0].index)
    plt.figure(figsize=(20, 10))
    ax = (dataWeek[column_name].groupby(dataWeek[column_name])
                         .count()).plot(kind="bar", color=color)
    ax.set_facecolor('#eeeeee')
    ax.set_xlabel("Hour of day")
    ax.set_xticks(np.arange(0, 75, 3))
    ax.set_xticklabels(np.arange(0, 24, 1)) 
    ax.set_ylabel("Number of trips")
    ax.set_title(title)
    plt.show()

def foliumMap():
    import folium
    from folium.plugins import MarkerCluster

    # Change colours

    def color_change(c):
        if(c < 15):
            return('red')
        elif(15 <= c < 30):
            return('orange')
        else:
            return('green')
    
    # Create base map
    
    Montreal = [45.508154, -73.587450]
    map = folium.Map(location = Montreal,
                     zoom_start = 12, 
                     tiles = "CartoDB positron")
    
    file_stations = 'Stations_2019.csv'
    stations = pd.read_csv(file_stations)
    lat = stations['latitude'].values
    lon = stations['longitude'].values
    name = stations['name'].values   
    
    marker_cluster = MarkerCluster(locations=[lat, lon]).add_to(map)        
    
    # Plot markers
       
    for _lat, _lon, _name in zip(lat, lon, name):
        folium.CircleMarker(location = [_lat, _lon], 
                            radius = 9, 
                            popup = _name, 
#                            fill_color = color_change(_cap), 
                            color = "gray", 
                            fill_opacity = 0.9).add_to(marker_cluster)
        
    f = 'map_station_cluster.html'
    map.save(f)
    
def flowCount(data):

    file_stations = 'Stations_2019.csv'
    stations = pd.read_csv(file_stations)
    lat = stations['latitude'].values
    lon = stations['longitude'].values
    name = stations['name'].values
    
    #select weekday
    data = data.drop(data[data['week_day_b'] == 1].index)
    #select morning
    data = data.drop(data[data['time_slice'].between(0,840)].index)
    print(data)    
    data_s = data.groupby('start_station_code').size().to_frame('departures_cnt').reset_index()
    data_e = data.groupby('end_station_code').size().to_frame('arrivals_cnt').reset_index()
  
    stations['departures'] = data_s['departures_cnt']
    stations['arrivals'] = data_e['arrivals_cnt']
    return stations
    

def densityMap(stations):

    Montreal = [45.508154, -73.587450]
    map = folium.Map(location = Montreal,
                zoom_start = 12,
                tiles = "CartoDB dark_matter")
    
    stations['net_departures'] = pd.Series( index=data.index)
    stations['net_departures'] = stations['departures'] - stations['arrivals']
    stations['radius'] = pd.Series( index=data.index)
    
    stations['radius'] = np.abs(stations['net_departures'])
    stations['radius'] = stations['radius'].astype(float)

    
    stations['color'] = '#E80018' # red 
    stations.loc[stations['net_departures'].between(-10000,0), 'color'] = '#00E85C' # green
     
    lat = stations['latitude'].values
    lon = stations['longitude'].values
    name = stations['name'].values
    rad = stations['radius'].values
    color = stations['color'].values
    net_dep = stations['net_departures']
  
    
    for _lat, _lon, _rad, _color, _name, _nd in zip(lat, lon, rad, color, name, net_dep):
        folium.CircleMarker(location = [_lat,_lon], 
                            radius = _rad/90,
                            color = _color,
                            tooltip = _name + " / net. dep:" +str(_nd),
                            fill = True).add_to(map)
  
    
    f = 'map_density.html'
    map.save(f)

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import folium
    from folium.plugins import MarkerCluster
    
    data = read()
    
    stations = flowCount(data)
    densityMap(stations)
    #foliumMap()
    #visualizePerDay(data, 'start_date')
    #visualizePerHourW(data, 'time_slice')
    #visualizePerHourE(data, 'time_slice')
    







