import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.preprocessing import PowerTransformer

def powerset(s):
	'''
	given input array, output the possible subset of the input array
	'''
	x = len(s)
	masks = [1 << i for i in range(x)]
	for i in range(1 << x):
		yield [ss for mask, ss in zip(masks, s) if i & mask]

mask = [0,1,2,3,4,5]
mask = list(powerset(mask))[1:]

# The train function input the filename and output the classifier
def train(filename):
    train_data = pd.read_csv(filename)
    
    #extract the features and labels 
    features = np.array(train_data.loc[:len(train_data),:'Feature_6'])
    #square the feature
    #features = np.square(features)
    label = np.array(train_data['Label']) 
    #tranform the features into a more gaussian way
    pt = PowerTransformer()
    pt.fit(features)
    feature_transformed = pt.transform(features)
    #find all the subset of features
    mask = [0,1,2,3,4,5]
    mask = list(powerset(mask))[1:]                              
    scores = []
    for i in mask:
        feature = feature_transformed[:, i]
        clf = GaussianNB()
        score = np.average(cross_val_score(clf, feature, label, cv = 10))
        scores.append(score)
        
    maxscore = max(scores) 
    print('maxscore is {}'.format(maxscore))
    selection = mask[scores.index(maxscore)]
    print('feature selected are {}'.format(selection))
    feature_select = feature_transformed[:,selection]
    nb = GaussianNB()
    nb.fit(feature_select,label)
    print('The average CV score is {}'.format(np.average(cross_val_score(nb, feature_select, label, cv = 10))))
    return nb

def predict(C, row):
    clf = C
    feature = np.array(row).reshape(1,-1)
    label_pred = clf.predict(row)
    return label_pred

nbclf = train('hw1_trainingset.csv')
df_test = pd.read_csv('hw1_testset.csv')
n = len(df_test)
feature_test = np.array(df_test.loc[:len(df_test),:'Feature_6'])
pt = PowerTransformer()
pt.fit(feature_test)
feature_test = pt.transform(feature_test)
label_predict = []
for i in range(n):
    fea = feature_test[i][[1,3,4]].reshape(1,-1)
    label_single = predict(nbclf, fea)[0]
    label_predict.append(label_single)
df_test['Label'] = label_predict
export_csv = df_test.to_csv ('prediction.csv', index = None, header=True)




