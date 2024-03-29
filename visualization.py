# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 14:44:46 2019

@author: greg0
"""
# Load journeys in a panda dataframe
def read():
    f = 'OD_2019-05.csv'
    data = pd.read_csv(f)
    
    #filtering out trips that were less tha
    data = data.drop(data[data['duration_sec'] < 120].index)
    
    #parsing through date data
    data['start_date'] = pd.to_datetime(data['start_date'], format='%Y-%m-%d', errors='coerce')
    data['end_date'] = pd.to_datetime(data['end_date'], format='%Y-%m-%d', errors='coerce')
    
    #adding a day number column to identify days of the week
    data['day_number'] = pd.to_datetime(data['start_date']).dt.dayofweek
    
    #adding a boolean week day column
    data['week_day_b'] = pd.Series( index=data.index)
    data['week_day_b'] = 1
    data.loc[data['day_number'].isin([5, 6]), 'week_day_b'] = 0
             
    #adding a column for to split the departures in 20 min groups
    data['time_slice'] = data['start_date'].dt.hour * 60 + np.floor(data['start_date'].dt.minute/20)*20
    pd.options.mode.chained_assignment = None
    data['time_slice'] = data['time_slice'].astype('int64')
    
    return data

#creates a bar chart with the departures per day
def visualizePerDay(data, column_name, color='#0000FF', title='Departure per day'):
  
    plt.figure(figsize=(20, 10))
    ax = (data[column_name].groupby(data[column_name].dt.day)
                         .count()).plot(kind="bar", color=color)
    ax.set_facecolor('#eeeeee')
    ax.set_xlabel("time of day")
    ax.set_ylabel("number of trips")
    ax.set_title(title)
    plt.show()
    
#creates a bar chart with the departures per hour during the weekend
def visualizePerHourEnd(data, column_name, color='#0000FF', title='Avreage Number of Trips Per Hour During the Weekend - Mai 2019'):
    
    #WEEKEND
    dataWeekend = data.drop(data[data['week_day_b'] == 1].index)
    plt.figure(figsize=(20, 10))
    ax = ((dataWeekend[column_name].groupby(dataWeekend[column_name].dt.hour)
                         .count())/8).plot(kind="bar", color=color)
   
    ax.set_xlabel("Hour")
    ax.set_ylabel("Number of Trips")
    ax.set_title(title)
    plt.rcParams.update({'font.size': 22})
    plt.show()

#creates a bar chart with the departures per hour during the week
def visualizePerHourWeek(data, column_name, color='#0000FF', title='Avreage Number of Trips Per Hour During the Week - Mai 2019'):
    
    dataWeek = data.drop(data[data['week_day_b'] == 0].index)
    plt.figure(figsize=(20, 10))
    ax = ((dataWeek[column_name].groupby(dataWeek[column_name].dt.hour)
                         .count())/23).plot(kind="bar", color=color)
   
    ax.set_xlabel("Hour")
    ax.set_ylabel("Number of Trips")
    ax.set_title(title)
    plt.rcParams.update({'font.size': 22})
    plt.show()

def foliumMap():
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
    
    # Plot markers for stations
    for _lat, _lon, _name in zip(lat, lon, name):
        folium.CircleMarker(location = [_lat, _lon], 
                            radius = 9, 
                            popup = _name,
                            color = "gray", 
                            fill_opacity = 0.9).add_to(marker_cluster)
        
    f = 'maps/map_station_cluster.html'
    map.save(f)
    
def flowCount(data, start_time, end_time):

    file_stations = 'Stations_2019.csv'
    stations = pd.read_csv(file_stations)
    
    #drop weekends
    data = data.drop(data[data['week_day_b'] == 0].index)
    
    #select time slot
    data = data.drop(data[data['time_slice'].between(0,start_time)].index)
    data = data.drop(data[data['time_slice'].between(end_time, 1420)].index)
    
    #agregate the number of departures per stations
    data_s = data.groupby('start_station_code').size().to_frame('departures_cnt').reset_index()
    data_s = data_s.rename(columns={'start_station_code':'Code'})
    data_e = data.groupby('end_station_code').size().to_frame('arrivals_cnt').reset_index()  
    data_e = data_e.rename(columns={'end_station_code':'Code'})
    
    #add a net departure column
    stations = pd.merge(stations, data_s, on='Code')
    stations = pd.merge(stations, data_e, on='Code')
    stations['net_departures'] = pd.Series( index=data.index)
    stations['net_departures'] = stations['departures_cnt'] - stations['arrivals_cnt']
    
    #replace stations with 0 net_departures by 1 to avoid calcultions using 0
    stations.loc[stations['net_departures'].eq(0), 'net_departures'] = 1
    
    return stations
    
def densityMap(stations):
    
    #generate a new map
    Montreal = [45.508154, -73.587450]
    map = folium.Map(location = Montreal,
                zoom_start = 12,
                tiles = "CartoDB positron")
    
    #calculate stations radius
    stations['radius'] = pd.Series( index=data.index)
    stations['radius'] = np.abs(stations['net_departures'])
    stations['radius'] = stations['radius'].astype(float)

    #set stations color
    stations['color'] = '#E80018' # red 
    stations.loc[stations['net_departures'].between(-10000,0), 'color'] = '#00E85C' # green
     
    lat = stations['latitude'].values
    lon = stations['longitude'].values
    name = stations['name'].values
    rad = stations['radius'].values
    color = stations['color'].values
    net_dep = stations['net_departures']
  
    #populate map
    for _lat, _lon, _rad, _color, _name, _nd in zip(lat, lon, rad, color, name, net_dep):
        folium.Circle(location = [_lat,_lon], 
                            radius = _rad/5,
                            color = _color,
                            tooltip = _name + " / net. dep:" +str(_nd),
                            fill = True).add_to(map)
  
    #save map
    f = 'maps/map_density_840_1260.html'
    map.save(f)

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import folium
    from folium.plugins import MarkerCluster
    
    data = read()
    #stations_flow = flowCount(data, 840, 1260)
    
    #densityMap(stations_flow)
    #foliumMap()
    #visualizePerDay(data, 'start_date')
    visualizePerHourWeek(data, 'start_date')
    visualizePerHourEnd(data, 'start_date')
    







