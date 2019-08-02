# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 14:56:28 2019
@author: greg0
"""

#read CSV file
def read_trips():
      
    data_files = glob.glob('data2018/OD_2018-*.csv')
    
    li = []
    
    for filename in data_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    data = pd.concat(li, axis=0, ignore_index=True)
    data['start_date'] = pd.to_datetime(data['start_date'], format='%Y-%m-%d', errors='coerce')
    data['end_date'] = pd.to_datetime(data['end_date'], format='%Y-%m-%d', errors='coerce')
    
    #filtering out trips that were less than 2 min
    data = data.drop(data[data['duration_sec'] < 120].index)
    
    #adding a day number column to identify days of the week
    data['day_number'] = pd.to_datetime(data['start_date']).dt.dayofweek
    data = data.drop(data[data['day_number'].isin([5, 6])].index)
    
    return data

#return an afinity matrix
def stations_connectivity(data):
    outbound = pd.crosstab(data['start_station_code'], data['end_station_code'])
    inbound = pd.crosstab(data['end_station_code'], data['start_station_code'])
    #using the sum gives us and undirected affinity
    #this makes the matrix symmetrical across the diagonal, required by spectral clustering
    connectivity = inbound + outbound
    #spectral clustering also requires the diagonal to be zero
    np.fill_diagonal(connectivity.values, 0)
    return connectivity

def cluster_labels_to_station_ids(connectivity, labels):
    no_clusters = len(set(labels))
    
    station_clusters = [ [] for n in range(0, no_clusters)]
    for idx, label in enumerate(labels):
        station = connectivity.columns[idx]
        station_clusters[label].append(station)
    
    #largest cluster first
    station_clusters = sorted(station_clusters, key=len, reverse=True)
    return station_clusters

#Perform clustering using spectral clustering
def cluster_spectral(data, n_clusters):
    connectivity =  stations_connectivity(data)
    cluster = sklearn.cluster.SpectralClustering(n_clusters=n_clusters, affinity ='precomputed')
    labels = cluster.fit_predict(connectivity)
    
    station_clusters = cluster_labels_to_station_ids(connectivity, labels)
    
    return station_clusters

def read_stations():
    file_stations = 'data2018/Stations_2018.csv'
    stations = pd.read_csv(file_stations)    
    return stations

def sations_clust(stations, clustered):
    
    stations['cluster'] = pd.Series( index=data.index)
    
    i = 0
    for list in clustered:
        for x in list:
            stations.loc[stations['code'] == x, 'cluster'] = i
        i = i + 1
    
    #color by cluster
    stations['color'] = pd.Series( index=data.index)
    stations.loc[stations['cluster'] == 0, 'color'] = "#0000FF" #blue
    stations.loc[stations['cluster'] == 1, 'color'] = "#FF0000" #red
    stations.loc[stations['cluster'] == 2, 'color'] = "#4B0082" #purple
    stations.loc[stations['cluster'] == 3, 'color'] = "#FF1493" #pink
    stations.loc[stations['cluster'] == 4, 'color'] = "#32CD32" #green
    stations.loc[stations['cluster'] == 5, 'color'] = "#FF4500" #orange
    stations.loc[stations['cluster'] == 6, 'color'] = "#FFFF00" #yellow
    stations.loc[stations['cluster'] == 7, 'color'] = "#800000" #brown
    stations.loc[stations['cluster'] == 8, 'color'] = "#000000" #black
    stations.loc[stations['cluster'] == 9, 'color'] = "#00FFFF" #aqua
    stations.loc[stations['cluster'] == 10, 'color'] = "#A9A9A9" #gray
    stations.loc[stations['cluster'] == 11, 'color'] = "#FFA07A" #salmon
    stations.loc[stations['cluster'] == 12, 'color'] = "#006400" #dark green
    
       
    return stations

def map_clustured(stations):
    # Create base map
    Montreal = [45.508154, -73.587450]
    map = folium.Map(location = Montreal,
                     zoom_start = 12, 
                     tiles = "CartoDB positron")
    
    lat = stations['latitude'].values
    lon = stations['longitude'].values
    name = stations['name'].values   
    color = stations['color'].values    
    
    # Plot markers for stations
    for _lat, _lon, _name, _color in zip(lat, lon, name, color):
        folium.Circle(location = [_lat, _lon], 
                            radius = 30, 
                            popup = _name,
                            color = _color).add_to(map)
        
    f = 'maps/map_station_clustered_10.html'
    map.save(f)
    
def cluster_trips(stations_clustered, data):
    
    data = pd.merge(data, stations_clustered[['code','cluster']], left_on='start_station_code', right_on='code')
    data = data.rename(columns={'cluster':'cluster_from'})
    
    data = pd.merge(data, stations_clustered[['code','cluster']], left_on='end_station_code', right_on='code')
    data = data.rename(columns={'cluster':'cluster_to'})
    
    data = data.groupby(['cluster_from','cluster_to']).size()
    data = data.to_frame('count').reset_index()
    
    data['cluster_from'] = data['cluster_from'].astype('int64')
    data['cluster_to'] = data['cluster_to'].astype('int64')
    
    return data

def to_tuples(data):
    
    subset = data[['cluster_from', 'cluster_to', 'count']]
    tuples = [tuple(x) for x in subset.values]
    
    return tuples

def create_graph_data(data):
    
    total = data['count'].sum();
    data['weight'] = pd.Series( index=data.index)
    data['weight'] = data['count']/total
    
    
    data = data.rename(columns={'cluster_from':'from'})
    data = data.rename(columns={'cluster_to':'to'})
    data = data.drop(data[data['weight'] < 0.005].index)
    return data
    

def create_graph(data):
    #data = data.drop(data[data['weight'] < 0.01].index)
    
    G = nx.from_pandas_edgelist(data, 'from', 'to', edge_attr=True, create_using=nx.MultiDiGraph())
    
    
    
    #color = ["#0000FF", "#FF0000", "#4B0082", "#FF1493", "#32CD32", "#FF4500", "#FFFF00", "#800000"]
    color = ["#0000FF", "#FF0000", "#4B0082", "#FF1493", "#32CD32", "#FF4500", "#FFFF00"]
    #nx.draw(G, with_labels=True, node_color=color, node_size=1000)
    nx.draw(G, with_labels=True, node_color=color, node_size=1000)
    plt.show()

def create_graphviz(data):
    G = Digraph(format='png')
    
    G.attr(rankdir='LR', size='10')
    G.attr('node', shape='circle')
    
    nodelist = []
    data = data.drop(['count'], axis=1)
    data['from'] =  pd.to_numeric(data['from'], downcast='integer')
    data['to'] =  pd.to_numeric(data['to'], downcast='integer')
    data['from'] =  data['from'].apply(str)
    data['to'] =  data['to'].apply(str)
    
    
            
    for idx, row in data.iterrows():
        node1, node2, weight = [str(i) for i in row]

        if node1 not in nodelist:
            G.node(node1)
            
            nodelist.append(node2)
        if node2 not in nodelist:
            G.node(node2)
            nodelist.append(node2)
        
        percent = float(weight)
        
        percent = percent*100
        
        percent = round(percent, 2)
        
        percent = str(percent)
        
        G.edge(node1,node2, label = (""+ percent +" %"))
    
    G.render('bixi_graph', view=True)
    
    
if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import glob
    import sklearn.cluster
    import networkx as nx
    import matplotlib.pyplot as plt
    import os
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    from graphviz import Digraph
    import folium
    
    data = read_trips()
    
    clustered = cluster_spectral(data, n_clusters=10)
    
    stations = read_stations()
    
    stations_clustered = sations_clust(stations, clustered)
    
    data = cluster_trips(stations_clustered, data)
    
    data = create_graph_data(data)
    
    create_graphviz(data)
    
    #map_clustured(stations_clustered)
    
    
