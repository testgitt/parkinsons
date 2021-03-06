from sklearn.cluster import KMeans
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import style
style.use("ggplot")
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from sklearn.cluster import KMeans
#from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

names= [ 'A' , 'B' ,'C' ,'D' ,'E' ,'F' ,'G' ,'H' ,'I','J','K','L','M' ,'N' ,'O','P','Q','R','S','T','U','V','W','X','Y','Z','AA','AB','AC']
train = pd.read_csv("mparktest2.csv", names=names)
features_one = train[[ "P","Q","R","S","T","B", "C","D","E","F","G","H","I","J","K","L","M","N","O","AC" ]].values
names = ['name','Fo','Fm','sd','Flo','Fhi', 'j','Jitter(Abs)','MDVP:RAP','MDVP:PPQ','Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','APQ11','SDDA','AC','NHR','HNR']
df = pd.DataFrame(features_one, columns= [ "P","Q","R","S","T","B", "C","D","E","F","G","H","I","J","K","L","M","N","O","AC" ])
features_one=pd.DataFrame(df[(df["AC"] == 1)])
#print(features_one)
X=pd.DataFrame(features_one, columns= [ "P","T","R","C","H","J","K","N" ])
#print(X)
n_samples, n_features = X.shape
y = train['AC'].values
n_digits = len(np.unique(y))
#labels = train.AC
#print(n_digits)
sample_size = 300

#print("n_digits: %d, \t n_samples %d, \t n_features %d"
 #     % (n_digits, n_samples, n_features))
np.random.seed(5)
test = pd.read_csv("info.csv" ,names=names)
test_features = test[[ "Fo","Fhi","sd","Jitter(Abs)","MDVP:Shimmer(dB)","Shimmer:APQ5","APQ11","HNR"]].values
# Initialize the model with 2 parameters -- number of clusters and random state.
kmeans = KMeans(n_clusters=3, random_state=1)
#print(kmeans)
# Get only the numeric columns from games.
good_columns = X._get_numeric_data()
# Fit the model using the good columns.
kmeans.fit(good_columns)
# Get the cluster assignments.
labels = kmeans.labels_
# Create a PCA model.
pca_2 = PCA(2)
# Fit the PCA model on the numeric columns from earlier.
plot_columns = pca_2.fit_transform(good_columns)
pred=kmeans.predict(test_features)
print('CONDITION OF THE DISEASE:')
#print(pred)
if(pred==0):
  print('low')
if(pred==1):
  print('low')
if(pred==2):
  print('high')
# Make a scatter plot of each game, shaded according to cluster assignment.
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
#plt.savefig('cluster.png')
# Show the plot.
#plt.show()
PassengerId =np.array(test["name"]).astype(str)
my_solutionc = pd.DataFrame(pred, PassengerId)
my_solutionc.to_csv("my_solution_onec.csv", mode='a')

#df=df['K Mean predicted label'] =kmeans.labels_
#print(df)


test = pd.read_csv("info.csv" ,names=names)
test_features = test[[ "Fo","Fhi","sd","Jitter(Abs)","MDVP:Shimmer(dB)","Shimmer:APQ5","APQ11","HNR"]].values

#kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
centroids=kmeans.cluster_centers_
labels=kmeans.labels_
#print(centroids)
#print('labels:')
#print(labels)
#colors = ["g.","r."]
#for i in range(len(X)):
 #   print("coordinate:",X[i], "label:", labels[i])
  #  plt.plot(X[i][0],X[i][1], colors[labels[i]],markersize=10)

#plt.scatter(centroids[:, 0],centroids[:, 1],markers="x",s=150,linewidth=5,zorder=10)
#plt.show()

pred=kmeans.predict(test_features)

print('the centers of the cluster');
centers=kmeans.cluster_centers_
print(centers)
plt.savefig('cluster.png')

