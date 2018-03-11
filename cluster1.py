from sklearn.cluster import KMeans
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from sklearn.cluster import KMeans
#from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

names= [ 'A' , 'B' ,'C' ,'D' ,'E' ,'F' ,'G' ,'H' ,'I','J','K','L','M' ,'N' ,'O','P','Q','R','S','T','U','V','W','X','Y','Z','AA','AB','AC']
train = pd.read_csv("mparktest.csv", names=names)
features_one = train[[ "P","S","T", "C","E","F","G","H","I","J","AC" ]].values
names = [ 'Fo', 'Fhi', 'Flo', 'j','Jitter(Abs)','MDVP:RAP','MDVP:PPQ','Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','NHR']
df = pd.DataFrame(features_one, columns= [ "P","S","T", "C","E","F","G","H","I","J","AC" ])
features_one=pd.DataFrame(df[(df["AC"] == 1)])
print(features_one)
features_two=pd.DataFrame(features_one, columns= ["P","S","T", "C","E","F","G","H","I","J" ])
print(features_two)
n_samples, n_features = features_two.shape
target = train['AC'].values
n_digits = len(np.unique(target))
labels = train.AC
print(n_digits)
sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))




test = pd.read_csv("info4.csv" ,names=names)
test_features = test[[ "Fo","Fhi", "Flo","Jitter(Abs)","MDVP:RAP","Jitter:DDP","MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3","Shimmer:APQ5"]].values

kmeans = KMeans(n_clusters=2, random_state=0).fit(features_two)
labels=kmeans.labels_
print(labels)
pred=kmeans.predict(test_features)
print(pred)
kmeans.cluster_centers_

