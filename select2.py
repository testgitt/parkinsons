import numpy as np
import urllib
import matplotlib.pyplot as plt
#import pandas
# url with dataset
#url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
# download the file
#raw_data = urllib.urlopen(url)
# load the CSV file as a numpy matrix
dataset = np.loadtxt('park.csv', delimiter=",",skiprows=1,usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17))
# separate the data from the target attributes
X = dataset[:,0:16]
y = dataset[:,16]
print(X)
print(y)
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X, y)
# display the relative importance of each attribute
print(model.feature_importances_)

