
#Exploratory Analysis of Geolocational Data
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import geopy

df = pd.read_csv('food_coded.csv')
df.head()

df.info()

df.describe()

"""Filling NaN data with the mode """

for column in df.columns:
    df[column].fillna(df[column].mode()[0], inplace=True)
df.head()

"""Creating a new dataframe with essential columns"""

df1=df[['cook','eating_out','employment','ethnic_food','exercise','fruit_day','income','on_off_campus','pay_meal_out','sports', 'veggies_day']]

print(df.columns.tolist())

df1.head()

df1.shape

"""#Removal of Outliers - Using Boxplots




"""

fig, ax = plt.subplots(figsize=(14, 8))
sns.boxplot(data=df1, ax = ax)

max_threshold = df1.quantile(0.95)
max_threshold

df1 = df1.drop(df1[df1['cook'] > 4].index)
df1 = df1.drop(df1[df1['eating_out'] > 4].index)
df1 = df1.drop(df1[df1['fruit_day'] < 3].index)
df1 = df1.drop(df1[df1['on_off_campus'] > 1].index)
df1 = df1.drop(df1[df1['pay_meal_out'] > 5].index)

df1.shape

df1.head()

fig, ax = plt.subplots(figsize=(14, 8))
sns.boxplot(data=df1, ax = ax)

df1 = df1.reset_index()

df1 = df1.drop(df1[df1['veggies_day'] < 2].index)

df1.head()

df1['index']

df1 = df1.drop(["index"], axis = 'columns')

df1.head()

fig, ax = plt.subplots(figsize=(14, 8))
sns.boxplot(data=df1, ax = ax)

df1 = df1.drop(df1[df1['veggies_day'] < 3].index)

fig, ax = plt.subplots(figsize=(14, 8))
sns.boxplot(data=df1, ax = ax)

df1.shape

df1.head()

df1 = df1.reset_index()

df1 = df1.drop(["index"], axis = 'columns')

"""The new refined dataset"""

df1.head()

"""#KMeans Clustering """

X = df1.copy()

X.head()

"""Elbow Method to find the number of clusters"""

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
  kmeans=KMeans(n_clusters=i,init = 'k-means++',random_state=42)
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('The number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans=KMeans(n_clusters=3, init='k-means++',random_state=42)
Y_kmeans=kmeans.fit_predict(X) 
print(Y_kmeans)

"""#Foursquare API - To get geolocational data"""

import json, requests
CLIENT_ID = 'XIIM5W1A0NKAFDSEAUJ34FHU0GHVJ5FUESJPXXPZ1PJEGKB0' # your Foursquare ID
CLIENT_SECRET = '0D52FZGNWAS5FLJ1Y5QOFYPAGM12CWG13NS4O2OEVSPT4YIE' # your Foursquare Secret
VERSION = '20200316'
LIMIT = 10000
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    13.0306, 77.5649,
    30000, 
    LIMIT)
data = requests.get(url).json()

data

"""Converting the Json Response to a Dataframe"""

dataframe = pd.json_normalize(data['response']['groups'][0]['items'])

dataframe.head(10)

"""Finding the number of restaurants and other amenities like gyms, parks in the given location (using lat and lng)"""

resta=[]
oth=[]
for lat,lng in zip(dataframe['venue.location.lat'],dataframe['venue.location.lng']):
    url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
      CLIENT_ID, 
      CLIENT_SECRET, 
      VERSION, 
      lat,lng,
      1000, 
      100)
    res = requests.get(url).json()
    venue = res['response']['groups'][0]['items']
    dataframe = pd.json_normalize(venue)
    df=dataframe['venue.categories']

    g=[]
    for i in range(0,df.size):
      g.append(df[i][0]['icon']['prefix'].find('food'))
    co=0
    for i in g:
      if i>1:
        co+=1
    resta.append(co)
    oth.append(len(g)-co)
dataframe

dataframe['restaurant'] = pd.Series(resta)
dataframe['others']=pd.Series(oth)

dataframe

dataframe

dataframe=dataframe[['venue.location.lat','venue.location.lng','restaurant','others','venue.location.address']]

dataframe.head(10)

dataframe = dataframe.dropna()

dataframe.isna().sum()

dataframe = dataframe.rename(columns={'venue.location.lat': 'lat', 'venue.location.lng': 'lng'})

dataframe.head()

X1 = dataframe.copy()
X1 = X1.drop(['venue.location.address'],axis =1)

X1.head(25)

"""#KMeans Clustering - Finding the Clusters and Centroids, using the new FourSquare Dataframe (X1)"""

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
  kmeans=KMeans(n_clusters=i,init = 'k-means++',random_state=42)
  kmeans.fit(X1)
  wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('The number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans=KMeans(n_clusters=5, init='k-means++',random_state=42)
X1['cluster']=kmeans.fit_predict(X1)

X1.head(25)

closest = kmeans.cluster_centers_
X1["centroids"] = 0
for i in range(len(closest)):
    X1["centroids"].iloc[i] = 1

X1[X1['centroids']==1]

X1['venue.location.formattedAddress'] = dataframe['venue.location.address']
X1.head()

"""Folium to plot the clusters on a map"""

import folium
m = folium.Map(location=[12, -122.6750])

city = "Bangalore"
## get location
locator = geopy.geocoders.Nominatim(user_agent="MyCoder")
location = locator.geocode(city)
print(location)
## keep latitude and longitude only
location = [location.latitude, location.longitude]
print("[lat, lng]:", location)

"""#Plotting the Clusters on a geographical location"""

import sklearn.preprocessing
import folium
x, y = "lat", "lng"
color = "cluster"
size = "restaurant"
popup = "venue.location.formattedAddress"
marker = "centroids"
data = X1.copy()
## create color column
lst_elements = sorted(list(X1[color].unique()))
lst_colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for i in 
              range(len(lst_elements))]
data["color"] = data[color].apply(lambda x: 
                lst_colors[lst_elements.index(x)])
## create size column (scaled)
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(3,15))
data["size"] = scaler.fit_transform(
               data[size].values.reshape(-1,1)).reshape(-1)
## initialize the map with the starting location
map_ = folium.Map(location=location, tiles="cartodbpositron",
                  zoom_start=11)
## add points
data.apply(lambda row: folium.CircleMarker(
           location=[row[x],row[y]], 
           color=row["color"], fill=True,popup=row[popup],
           radius=row["size"]).add_to(map_), axis=1)
## add html legend
legend_html = """<div style="position:fixed; bottom:10px; left:10px; border:2px solid black; z-index:9999; font-size:14px;">&nbsp;<b>"""+color+""":</b><br>"""
for i in lst_elements:
     legend_html = legend_html+"""&nbsp;<i class="fa fa-circle 
     fa-1x" style="color:"""+lst_colors[lst_elements.index(i)]+"""">
     </i>&nbsp;"""+str(i)+"""<br>"""
legend_html = legend_html+"""</div>"""
map_.get_root().html.add_child(folium.Element(legend_html))
## add centroids marker
lst_elements = sorted(list(X1[marker].unique()))
data[data[marker]==1].apply(lambda row: 
           folium.Marker(location=[row[x],row[y]], 
           draggable=False,  popup=row[popup] ,       
           icon=folium.Icon(color="black")).add_to(map_), axis=1)
## plot the map
map_