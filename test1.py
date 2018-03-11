import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import tree
import cgi,cgitb
import os
import sys
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#data =sys.argv[1]
#print(data)

names= [ 'A' , 'B' ,'C' ,'D' ,'E' ,'F' ,'G' ,'H' ,'I','J','K','L','M' ,'N' ,'O','P','Q','R','S','T','U','V','W','X','Y','Z','AA','AB','AC']
train = pd.read_csv("mparktest.csv", names=names)
#names = [ 'Fo', 'Fhi', 'Flo', 'j%','Jitter(Abs)','MDVP:RAP','MDVP:PPQ','Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','NHR']
#names = ['name','Fo','Fm','sd','Flo','Fhi', 'j','Jitter(Abs)','MDVP:RAP','MDVP:PPQ','Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','APQ11','SDDA','AC','NHR','HNR']
names = ['A' , 'B' ,'C' ,'D' ,'E' ,'F' ,'G' ,'H' ,'I','J','K','L','M' ,'N' ,'O','P','Q','R','S','T','U','V','W','X','Y','Z','AA']
test = pd.read_csv("info.csv" ,names=names)
# Print the train data to see the available features
#print(train)
#name=test['name'].values
#nam=str(name)
#print('nam')
# Create the target and features numpy arrays: target, features_one
#target = train["status"].values
target = train['AC'].values
#features_one = train[[ "MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)","j" ,"MDVP:Jitter(Abs)","MDVP:RAP","MDVP:PPQ","Jitter:DDP","MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3","Shimmer:APQ5","NHR" ]].values
features_one = train[[ "B" ,"C" ,"D" ,"E" ,"F" ,"G" ,"H" ,"I","J","K","L","M" ,"N" ,"O","P","Q","R","S","T","U","V","W","X","Y","Z","AA" ]].values
#test.j[1] = train.j[190]
#print (test.j)

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# Look at the importance and score of the included features
print('feature importances:')
print(my_tree_one.feature_importances_)
print('accuracy:')
#print(PHP_EOL)
print(my_tree_one.score(features_one, target))
test_features = test[[ 'B' ,'C' ,'D' ,'E' ,'F' ,'G' ,'H' ,'I','J','K','L','M' ,'N' ,'O','P','Q','R','S','T','U','V','W','X','Y','Z','AA' ]].values

# Make your prediction using the test set and print them.
my_prediction = my_tree_one.predict(test_features)

#print('prediction: \n',my_prediction )
#print(my_prediction)
print('\n')
print('RESULTS:')
if(my_prediction==0):
    print('NO PARKINSONS DISEASE')
if(my_prediction==1):
    print('there is a slight chance')
tree.export_graphviz(my_tree_one,out_file='tree.dot')   

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["A"]).astype(str)
my_solution = pd.DataFrame(my_prediction,PassengerId)
#print('name, prediction: \n', my_solution)
#print(my_solution)

# Check that your data frame has 418 entries
#print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_one.csv", mode='a')
features_two = train[[ "C" ,"D" ,"E" ,"F" ,"G" ,"H" ,"I","J","K","L","M","P","Q","R","S","T","U","V","X","Y"]].values
#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
my_tree_two = my_tree_two.fit(features_two, target)
test_features = test[[  'C' ,'D' ,'E' ,'F' ,'G' ,'H' ,'I','J','K','L','M','P','Q','R','S','T','U','V','X','Y']].values
my_prediction = my_tree_two.predict(test_features)
#print('prediction: \n',my_prediction )
#Print the score of the new decison tree
#print(my_prediction)
print('\n')
print('accuracy after feature selection and pruning:')
print(my_tree_two.score(features_two, target))
#print(my_prediction)
if(my_prediction==0):
    print('NO PARKINSONS DISEASE')
if(my_prediction==1):
    print('there is a slight chance developing disease')
tree.export_graphviz(my_tree_two,out_file='tree1.dot')   

  
