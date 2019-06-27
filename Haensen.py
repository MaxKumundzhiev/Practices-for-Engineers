# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

########################################## The Work Plan #####################################################
# ______________________________________________________________________________________________________________________
#
# Part 1
# Untuned solution:
#
# 1) Check the target - the number of classes - understand which type of task we have
#     a) if numeric - okay
#     b) if categorical - one hot encoding or dummy function
# 2) The target - check balanced / imbalanced situation + visualisation
# 3) Check the types of features
#     a) if numeric - okay
#     b) if categorical - one hot encoding or dummy function
# 4) Check the distribution of data<br>
#     a) if distribution is high - normalize / standardize
#     b) if distribution is NOT high(all the random values is near with mean) - okay
# 5) Split data (lots of ways(Validation, CrossValidation))
# 6) Straight solution with few models, define benchmark
#
# ______________________________________________________________________________________________________________________
#
# Part 2
# Tuned solution:
#
# 1) Implement Feature Selection and define features with the most Information Gain (reduce the number of features)
# 2) Split data (lots of ways(Validation, CrossValidation))
# 3) Straight solution with few models, define benchmark for modified data
# 4) Compare results

########################################## Import Libraries #####################################################

#EDA
import time

import pandas as pd
import numpy as np
import sklearn
import scipy

#Plotting
import matplotlib.pyplot as plt
import seaborn as sns

#Ignore warnings
import warnings
warnings.simplefilter('ignore')

#Preproccesing
from sklearn import preprocessing
from sklearn.model_selection import KFold, cross_val_score

#Metrics
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

#Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

#Feature selecting
from boruta import BorutaPy


########################################## Part 1 #####################################################
########################################## EDA + Visualisation + Fitting models.  #####################################################
print('Part 1')
#Read the data

data = pd.read_csv('/Users/macbook/Desktop/Haensen/sample.csv', header = None)

print('''1) Check the target - the number of classes - understand which type of task we have
a) if numeric - okay
b) if categorical - one hot encoding or dummy function''')

print(data.head())

print('Now, we can assume, that our task is MultiClassification task')

print('Rename target and encode it')
data = data.rename(columns=({295: 'y'}))

dic = {'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3, 'E' : 4}
data['y'] = data['y'].map(dic)

print(data.head())

print('2) The target - check balanced / imbalanced situation + visualisstion')

print(data['y'].value_counts(normalize = True), 'The target - check balanced / imbalanced case')
print('We can assume, that we have imbalanced classes')


# Distribution of target value
# On the histogramm we see problem with imbalanced classes better

#Normed - applies when multiple classes are plotted
plt.title('Distribution of target value on Histogram')
plt.hist(data['y'], normed = True)
plt.show()

#Distribution of target value
plt.title('Distribution of target value')
sns.distplot(data['y'], kde=True, rug=False)
plt.show()

#Correlation matrix of all the features
plt.title('Matrix of Correlation for all features')
corr_matrix = data.corr()
sns.heatmap(corr_matrix)
plt.show()

print('''\n It looks a little bit messy(correlation heatmap), but also can tell that lots of values are hight correlated(near with + 1.0).
It means that we can remove one from the boundary of them. But for now we will continue with situation in order to make a benchmark.''')

print('\n3) Check the types of features:')

print(data.info())

print('''\nWe can see that we have all the numerical values (int64 and float64).''')

print('''\n4) Check the distribution of data
a) if distribution is hight - normalize / standartize 
b) if distribution is NOT hight(all the random values is near with mean) - okay''')

print(data.describe(include='all').T.head())

print('\n', data.dtypes.value_counts())

print('''\nWe can see, that data has high values range (check the mean of features or min and max, etc),
,it means we should normalize / standartize data''')

print('\nNormalization')

#Split on X and y with normalization
y = data['y'].values
X_normalized = data.drop('y', axis = 1)
X_normalized = pd.DataFrame(data = preprocessing.normalize(data))


print('\nHave a look on a Normalized distribution')
print(X_normalized.describe(include='all').T.head())


print('''\n5) Split data for fitting(lots of ways(Validation, CrossValidation))
6) Straight solution with few models, also will define the benchmark''')

print('''\nDefine the Kfold CrossValidation''')
kf = KFold(n_splits=5, random_state=42, shuffle=True)

#kf.get_n_splits(X)
print(kf)

print('\n Fitting and evaluating differrent models on validation data: ')

########################################## KNN MODEL #####################################################

print('\n KNN')
print('\n For KNN we will try to validate with number of neighbours in the range from 1 upto 50 ')

def get_auc_knn_valid(X, y, neighbours_number, seed):
    #%time
    res = []
    for num_neighb in range(1,neighbours_number):
        knn = KNeighborsClassifier(n_neighbors=num_neighb, n_jobs = -1)
        estimator = knn
        prediction = cross_val_score(estimator, X, y, scoring='accuracy', cv = kf)
        res.append([prediction.mean(), num_neighb])
        #print(f'num_k %d'%num_neighb, prediction.mean())
        return(max(res))


print('KNN: ', get_auc_knn_valid(X_normalized, y, neighbours_number = 51, seed = 42))


########################################## Logistic Regression MODEL #####################################################
print('\n Logistic Regression')
print('''\n For Logistic Regression we will validate with 3 different Regularization parameters(The best one in our case is: C = 1.0)
multi_class = 'multinomial' - because we have Multiclassification Task''')

C = 1.0
#C = 1e4
#C = 1e-2

def get_auc_lr_valid(X, y, C = C, seed = 42):
    #%time
    logit = LogisticRegression(n_jobs = -1, random_state = seed, C = C, multi_class='multinomial', solver='lbfgs')
    #logit = LogisticRegression(n_jobs = -1, random_state = seed, C = C, multi_class='ovr', solver='lbfgs')
    estimator = logit
    prediction = cross_val_score(estimator, X_normalized, y, scoring='accuracy', cv = kf)
    return(prediction.mean())


print('\nLogistic regression: ', get_auc_lr_valid(X_normalized, y, C = 1.0, seed = 42))


########################################## SGDClassifier MODEL #####################################################
print('\nSGDClassifier')
print('''\n For SGDClassifier we will validate with loss = 'log', learning_rate='adaptive', eta0 = 0.05
and peanlty = 'l2'(squared magnitude), because our data is not extremly noisy, otherwise, we will use l1, because there we have module of sum''')

def get_auc_sgd_valid(X, y, seed = 42):
    #%time
    #sgd_logit = sklearn.linear_model.SGDClassifier(loss = 'log', random_state = seed, n_jobs = -1)
    sgd_logit = sklearn.linear_model.SGDClassifier(loss = 'log', learning_rate='adaptive', eta0 = 0.05,
                                              penalty = 'l2',  random_state = 42, n_jobs = -1)
    estimator = sgd_logit
    prediction = cross_val_score(estimator, X, y, scoring='accuracy', cv = kf)
    return(prediction.mean())


print('\nSGDClassifier: ', get_auc_sgd_valid(X_normalized, y, seed=42))

########################################## RandomForestClassifier MODEL #####################################################
print('\nRandomForestClassifier')
print('''\n For RandomForestClassifierwe will validate on 100 different DecisionTrees with max_depth = 2''')

def get_auc_rnd_forest_valid(X, y, seed = 42):
    #%time
    rand_forest = RandomForestClassifier(n_estimators = 100, max_depth=2, random_state = seed)
    estimator = rand_forest
    prediction = cross_val_score(estimator, X, y, scoring='accuracy', cv = kf)
    return(prediction.mean())


print('\nRandomForestClassifier: ', get_auc_rnd_forest_valid(X_normalized, y, seed = 42))

########################################## GradientBoostingClassifier MODEL #####################################################
print('\nGradientBoostingClassifier')
print('''\n For GradientBoostingClassifier will validate with 20 different learning rate values(from 0.05 up to 0.2) and
from 40 up to 70 estimators''')


def get_auc_gbm_valid(X, y):
    #% time
    plt.figure (figsize=(10, 5))
    plt.title ('Gradient Boosting model')
    result = []
    for lr in np.linspace (0.05, 0.2, 20):
        for n_est in range (40, 70):
            GBM_model = GradientBoostingClassifier(learning_rate=lr, n_estimators=n_est,
                                                   min_samples_leaf=50, min_samples_split=500, max_depth=8,
                                                   max_features='sqrt')

    estimator = GBM_model
    prediction = cross_val_score (estimator, X, y, scoring='accuracy', cv=kf)
    plt.plot(prediction)
    plt.show()
    result.append (prediction.mean())
    return result


print('\nGradientBoostingClassifier: ', get_auc_gbm_valid (X_normalized, y))


print('''\nFor now, we can assume, that the best approach was with GradienBoostingMachine. 
Let's define the benchmark as ≈0.92 (Solution without Feature Selection''')

########################################## Part 2  #####################################################
########################################## Feature Selection  #####################################################
print('\nPart 2')
print('''For now we will try decrease the number of features, finding the most informative of them.
We will do it using Boruta approach''')
print('\nBoruta Selector')

print('\nDefine the Boruta Selector and look at the parametrs: ')
rfc = RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight='balanced', max_depth=10)
boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=2, random_state=42)
print('\n', boruta_selector.get_params)

########################################## Boruta Selector  #####################################################
#%time
print(boruta_selector.fit(X_normalized.values, y))

print('''\nAfter the Selector executed, we will create a new DataFrame, where we will have ranked features.''')
feature_df = pd.DataFrame(X_normalized.columns.tolist(), columns=['features'])
feature_df['rank'] = boruta_selector.ranking_
feature_df = feature_df.sort_values('rank', ascending=True).reset_index(drop=True)
print('\n', feature_df[:20])


# number of selected features
print ('\n Number of selected features: %d' %boruta_selector.n_features_)

feature_df = pd.DataFrame(X_normalized.columns.tolist(), columns=['features'])
feature_df['rank'] = boruta_selector.ranking_
feature_df = feature_df.sort_values('rank', ascending=True).reset_index(drop=True)

print ('\n Top %d features:' % boruta_selector.n_features_)
print (feature_df.head(boruta_selector.n_features_))

feature_df.to_csv('boruta-feature-ranking.csv', index=False)

# check ranking of features
print ('\n Feature ranking:')
print (boruta_selector.ranking_)


print('\n After we will defined DataFrame with selected(with the most informative gain) features from Boruta Selector')

X_filtered = pd.DataFrame(data = X_normalized, columns = X_normalized.columns[boruta_selector.support_])
print(X_filtered.head())


print('\nAfter we again will look at heatmap(correlation matrix) of DataFrame which we just created using selected features')

#Plot a correlation matrix for X_filtered
def corr_matrix(data):
    plt.title('Matrix of Correlation for Selected Features')
    sns.set(style="white")

    # Compute the correlation matrix
    corr = data.corr ()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from (mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()


corr_matrix(X_filtered)

print('''\nWe can consider that almost all the values are not high correlated. At least few ones are, but we will try to fit models. <br>
In the case of two normal or almost normal values, the correlation coefficient between them can be used as a measure of interdependence 
and this is confirmed by many practical results.
However, when interpreting “interdependence”, the following difficulties are often encountered: if one quantity is correlated with another, 
this can only be a reflection of the fact that both of them are correlated with some third quantity or with a set of quantities that, 
roughly speaking, remain behind the frame and not entered into the model.''')

########################################## Fitting models with selected features  #####################################################
print('\n Fitting and evaluating different models on validation data(Modified data), algorithms are with the same hyperparametrs as before')

print('\nKNN')
print('\nKNN: ', get_auc_knn_valid(X_filtered, y, neighbours_number = 51, seed = 42))

print('\nLogistic Regression')
print('\nLogistic Regression: ', get_auc_lr_valid(X_filtered, y, C = 1.0, seed = 42))

print('\nSGDClassifier')
print('\nSGDClassifier: ', get_auc_sgd_valid(X_filtered, y, seed=42))

print('\nRandomForestClassifier')
print('\nRandomForestClassifier: ', get_auc_rnd_forest_valid(X_filtered, y, seed = 42))

print('\nGradientBoostingClassifier')
print('\nGradientBoostingClassifier: ', get_auc_gbm_valid(X_filtered, y))


print('''After Feature Selection we found 14 features with the highest Information Gain(using Boruta approach).
With Boruta and GradientBoostingClassifier(with the same hypeparameters) we increased our accuracy and got ≈0.99''')






