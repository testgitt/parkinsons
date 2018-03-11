import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import tree
train = pd.read_csv("park1.csv")
names = [ 'F0', 'Flo','Jitter(Abs)','MDVP:Shimmer(dB)','Shimmer:APQ5','HNR']
test = pd.read_csv("info.csv" ,names=names)
# Print the train data to see the available features
print(train)

# Create the target and features numpy arrays: target, features_one
target = train["status"].values
features_one = train[["MDVP:Fo(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(Abs)", "MDVP:Shimmer(dB)", "Shimmer:APQ5", "HNR"]].values

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# Look at the importance and score of the included features
print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one, target))
test_features = test[["F0", "Flo","Jitter(Abs)","MDVP:Shimmer(dB)","Shimmer:APQ5","HNR"]].values

# Make your prediction using the test set and print them.
my_prediction = my_tree_one.predict(test_features)
print('prediction: \n',my_prediction )
print(my_prediction)
tree.export_graphviz(my_tree_one,out_file='tree.dot')   

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
#PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, columns = ["Survived"])
print('prediction: \n', my_solution)
print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_one.csv")
#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
my_tree_two = my_tree_two.fit(features_one, target)

#Print the score of the new decison tree
print(my_tree_two.score(features_one, target))
