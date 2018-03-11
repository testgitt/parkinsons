#import pandas as pandas
# Load libraries
import pandas as pd
import numpy as np
#from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
#import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
from sklearn import tree
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# Load dataset

#names = ['name', 'F0', 'Fhi', 'Flo', 'j%','Jitter(Abs)','MDVP:RAP','MDVP:PPQ','Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA','NHR','HNR','status']
#dataset = pandas.read_csv('park2.csv', names=names, usecols=['F0', 'Flo','Jitter(Abs)','MDVP:Shimmer(dB)','Shimmer:APQ5','HNR','status'])
names= [ 'A' , 'B' ,'C' ,'D' ,'E' ,'F' ,'G' ,'H' ,'I','J','K','L','M' ,'N' ,'O','P','Q','R','S','T','U','V','W','X','Y','Z','AA','AB','AC']
#dataset = pd.read_csv("mparktest.csv", names=names,usecols=['B' ,'C' ,'D' ,'E' ,'F' ,'G' ,'H' ,'I','J','K','L','M' ,'N' ,'O','P','Q','R','S','T','U','V','W','X','Y','Z','AA','AC'])
dataset = pd.read_csv("mparktest.csv", names=names,usecols=['B' ,'C' ,'D' ,'E' ,'F' ,'G' ,'H' ,'I','J','K','L','N' ,'O','P','S','T','AB','AC'])
nam= ['name',  'j%','Jitter(Abs)','MDVP:RAP','MDVP:PPQ','Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA','ac','NHR','HNR','fo','fm','sd','flo','fhi','u','v','w','x','y','z','aa','ab','class']
dataset1 = pd.read_csv("mparktest.csv", names=nam,usecols=['Jitter:DDP','Shimmer:APQ5','MDVP:APQ','fo','flo','fhi','class' ])

#dataset = dataset1[[  'B' ,'C' ,'D' ,'E' ,'F' ,'G' ,'H' ,'I','J','K','L','P','S','T','AC']].values]
# shape
print('Dataset Shape')
print(dataset.shape)
# head
#print(dataset.head(20))
# descriptions
#print(dataset.describe())
# class distribution
#print(dataset.groupby('class').size())
# box and whisker plots
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()
# histograms
dataset1.hist()
plt.savefig('hist.png')
#plt.show()
#plt.hist(dataset["AC"])
#plt.show()
# scatter plot matrix
#scatter_matrix(dataset)
#plt.show()
# Split-out validation dataset
array = dataset.values
X = array[:,0:17]
Y = array[:,17]
validation_size = 0.30
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
# Test options and evaluation metric
seed = 4
scoring = 'accuracy'
print('CHECKING ACCURACY OF ALL THE ALGORITHMS THROUGH CROSS VALIDATION')
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print('accuracy for KNN')
print(accuracy_score(Y_validation, predictions))
print('confusion matrix for KNN')
print(confusion_matrix(Y_validation, predictions))
print('classification report  for KNN')
print(classification_report(Y_validation, predictions))

lr = LogisticRegression()
lr.fit(X_train, Y_train)
predictions = lr.predict(X_validation)
print('accuracy for LOGISTIC REGRESSION')
print(accuracy_score(Y_validation, predictions))
print('confusion matrix for LOGISTIC REGRESSION')
print(confusion_matrix(Y_validation, predictions))
print('classification report  for LOGISTIC REGRESSION')
print(classification_report(Y_validation, predictions))
#print(predictions)
cr = DecisionTreeClassifier()
cr.fit(X_train, Y_train)
predictions = cr.predict(X_validation)
print('accuracy for DECISION TREE')
print(accuracy_score(Y_validation, predictions))
print('confusion matrix for DECISION TREE')
print(confusion_matrix(Y_validation, predictions))
print('classification report  for DECISION TREE')
print(classification_report(Y_validation, predictions))
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)
cr = LinearDiscriminantAnalysis()
cr.fit(X_train, Y_train)
predictions = cr.predict(X_validation)
print('accuracy for LDA')
print(accuracy_score(Y_validation, predictions))
print('confusion matrix for LDA')
print(confusion_matrix(Y_validation, predictions))
print('classification report  for LDA')
print(classification_report(Y_validation, predictions))

clf = GaussianNB()
clf = clf.fit(X_train, Y_train)
#cr = DecisionTreeClassifier()
#cr.fit(X_train, Y_train)
predictions = clf.predict(X_validation)
print('accuracy for  NB')
print(accuracy_score(Y_validation, predictions))
print('confusion matrix for NB')
print(confusion_matrix(Y_validation, predictions))
print('classification report  for NB')
print(classification_report(Y_validation, predictions))


clf = SVC()
clf = clf.fit(X_train, Y_train)
#cr = DecisionTreeClassifier()
#cr.fit(X_train, Y_train)
predictions = clf.predict(X_validation)
print('accuracy for  SVM')
print(accuracy_score(Y_validation, predictions))
print('confusion matrix for SVM')
print(confusion_matrix(Y_validation, predictions))
print('classification report  for SVM')
print(classification_report(Y_validation, predictions))


# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.savefig('algcomp.png')
tree.export_graphviz(clf,out_file='tree.dot')
#tree.savefig('tree.dot')                
X, y = make_classification(200, 2, 2, 0, weights=[.5, .5], random_state=15)
clf = LogisticRegression().fit(X[:100], y[:100])
xx, yy = np.mgrid[-5:5:.01, -5:5:.01]
grid = np.c_[xx.ravel(), yy.ravel()]
probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
f, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
                      vmin=0, vmax=1)
ax_c = f.colorbar(contour)
ax_c.set_label("$P(y = 1)$")
ax_c.set_ticks([0, .25, .5, .75, 1])

ax.scatter(X[100:,0], X[100:, 1], c=y[100:], s=50,
           cmap="RdBu", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)

ax.set(aspect="equal",
       xlim=(-5, 5), ylim=(-5, 5),
       xlabel="$X_1$", ylabel="$X_2$")
f, ax = plt.subplots(figsize=(8, 6))
ax.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.6)

ax.scatter(X[100:,0], X[100:, 1], c=y[100:], s=50,
           cmap="RdBu", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)

ax.set(aspect="equal",
       xlim=(-5, 5), ylim=(-5, 5),
       xlabel="$X_1$", ylabel="$X_2$")

plt.savefig('lr');
#plt.show()

