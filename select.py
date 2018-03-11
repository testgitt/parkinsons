import pandas as pd
    

# load X and y
X = pd.read_csv('park.csv', index_col=0).values
from sklearn.feature_selection import VarianceThreshold
 
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(X)
