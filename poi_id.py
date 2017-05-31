#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### dataset background
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
### Store to my_dataset for easy export below.
my_dataset = data_dict
### dataset exploration ####
### number of entries ###
print "Number of entries= ",len(my_dataset)
### number of features ###
print "Number of features= ",len(my_dataset["HANNON KEVIN P"])
### number of poi's###
counter=0
for name in my_dataset.keys():
    if my_dataset[name]['poi']== True:
        counter += 1
        
print "Number of poi's in dataset= ",counter
### actual number of poi's###
import csv              
count = 0
with open("../final_project/poi_names.txt", "r") as na:
 names = csv.DictReader(na)
 for i in names:
  count = count+1
print "Actual total number of poi's = ",count
########### DATASET EXPLORATION ##############
f_list=[]
print "*************FEATURES***************"
for key, value in my_dataset["METTS MARK"].iteritems() :
    print key
    f_list.append(key)
print "************************************"
print "*********** Features with lots of missing data ********"
### MISSING DATA (>60) ###
for feature in f_list:
    count = 0
    for person in my_dataset:
        if my_dataset[person][feature] == 'NaN':
            count+=1
    if count> 60:
        print "# missing values in", feature ," = ", count     
### proportion of missing values 0.4 ###
for feature in f_list:
    count = 0
    for person in my_dataset:
        if my_dataset[person][feature] == 'NaN':
            count+=1
    if float(count)/float(146) > 0.4 :
        print "proportion of missing values in", feature ," = ", float(count)/float(146) 
print "*********** Features with less missing data ***********"
### MISSING DATA (<60) ###
for feature in f_list:
    count = 0
    for person in my_dataset:
        if my_dataset[person][feature] == 'NaN':
            count+=1
    if count< 60:
        print "# missing values in", feature ," = ", count     
### proportion of missing vlues <0.4 ###
for feature in f_list:
    count = 0
    for person in my_dataset:
        if my_dataset[person][feature] == 'NaN':
            count+=1
    if float(count)/float(146) < 0.4 :
        print "proportion of missing values in", feature ," = ", float(count)/float(146)
### Task 2: Remove outliers

import matplotlib.pyplot as plt    #############3  import statement ############
#### plotting salary vs expenses to see outliers ####
features = ["salary", "expenses"]
data = featureFormat(my_dataset, features,remove_NaN=True)
for n in data:
    salary = n[0]
    expense = n[1]
    plt.scatter( salary, expense )

plt.xlabel("salary")
plt.ylabel("expenses")
plt.show()

#### DETECTING OUTLIER -finding dictionary key###
max_value = float("-inf")

for k, v in my_dataset.iteritems():
    if v["salary"] != "NaN":
        if v["salary"] > max_value:
            max_value = v["salary"]
print "max_value for salary is ",max_value

for name in my_dataset.keys():
    if my_dataset[name]['salary'] == 26704229:
        print "Name of the outlier is",name
### REMOVING OUTLIER ###
del my_dataset['TOTAL'] 
### replotting ###
features = ["salary", "expenses"]
data = featureFormat(my_dataset, features,remove_NaN=True)
for n in data:
    salary = n[0]
    expense = n[1]
    plt.scatter( salary, expense )

plt.xlabel("salary")
plt.ylabel("expenses")
plt.show()
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list_full=['poi','to_messages', 'deferral_payments', 'expenses', 'deferred_income',  
                    'long_term_incentive', 'restricted_stock_deferred', 'shared_receipt_with_poi', 
                    'loan_advances', 'from_messages', 'other', 'director_fees', 'bonus', 'total_stock_value', 
                    'from_poi_to_this_person', 'from_this_person_to_poi', 'restricted_stock', 'salary', 'total_payments',
                    'exercised_stock_options']


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list_full, sort_keys = True)
labels, features = targetFeatureSplit(data)

### checking important features ####
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

clf=clf.fit(features,labels) ###fitting tree classifier##
most_important = max(clf.feature_importances_) ### finding important features###
print most_important
importances = clf.feature_importances_

import numpy as np
importances = np.array(importances)

### displaying the feature importances to select best features for poi identification###
print 'Feature Ranking: '
for i in range(19):
    print "{} feature {} ({})".format(i+1,features_list_full[i+1],importances[i])

### finalisisng features ####
features_list=['poi','exercised_stock_options','restricted_stock','total_payments','bonus','from_poi_to_this_person',
               'shared_receipt_with_poi','other','salary','expenses']
    
### calculating ranges for finalised features###
for feature in features_list:
    max_value = float("-inf")
    min_value = float("inf")
    for k in my_dataset:
        if my_dataset[k][feature] != "NaN":
            if my_dataset[k][feature] > max_value:
                max_value = my_dataset[k][feature]
            if my_dataset[k][feature] < min_value:
                min_value = my_dataset[k][feature]

    
    print "range: ", max_value-min_value,"in",feature

### scaling required which will be employed while using appropriate algorithm ###
### Task 3: Create new feature(s)

### feature engineering ###
### creating two new features ###
def Fraction( poi_messages, all_messages ):
    fraction = 0.
    if poi_messages != 'NaN' and  all_messages != 'NaN':
        fraction = float(poi_messages)/float(all_messages)
    return fraction



for name in my_dataset:
    
    from_poi_to_this_person = my_dataset[name]["from_poi_to_this_person"]
    to_messages = my_dataset[name]["to_messages"]
    fraction_from_poi = Fraction( from_poi_to_this_person, to_messages )
    my_dataset[name]["fraction_from_poi"] = fraction_from_poi


    from_this_person_to_poi = my_dataset[name]["from_this_person_to_poi"]
    from_messages = my_dataset[name]["from_messages"]
    fraction_to_poi = Fraction( from_this_person_to_poi, from_messages )

    my_dataset[name]["fraction_to_poi"] = fraction_to_poi



##### adding the new features to the selected features###
features_list = features_list + ["fraction_from_poi", "fraction_to_poi"]

### visualising new features ###
for key in my_dataset:
    plt.scatter( my_dataset[key]['fraction_to_poi'], my_dataset[key]['fraction_from_poi'], color = 'b')
    if my_dataset[key]['poi'] == True:
        plt.scatter(my_dataset[key]['fraction_to_poi'], my_dataset[key]['fraction_from_poi'], color='r', marker="*")
plt.ylabel('fraction of emails of this person to POI')
plt.xlabel('fraction of emails of this person from POI')
plt.show()

### checking importance ##
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### checking important features ####
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

clf=clf.fit(features,labels) ###fitting tree classifier##
most_important = max(clf.feature_importances_) ### finding important features###
print most_important
importances = clf.feature_importances_

import numpy as np
importances = np.array(importances)

### displaying the feature importances to select best features for poi identification###
print 'Feature Ranking: '
for i in range(11):
    print "{} feature {} ({})".format(i+1,features_list[i+1],importances[i])
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
###############################################################################

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Split into a training and testing set
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.1, random_state=42)

### deploying RandomForest Classifier on data and calling test_classifier function##
from tester import test_classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
print "\nRandom Forest"
test_classifier(clf, my_dataset, features_list_full)

### deploying AdaBoost Classifier on data and calling test_classifier function##
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier()
print "\n\nAdaboost"
test_classifier(clf, my_dataset, features_list)

### deploying DecisionTree Classifier on data and calling test_classifier function##
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
print "\n\nDecision Tree Classifier"
test_classifier(clf, my_dataset, features_list)

### deploying RandomForest Classifier on data and calling test_classifier function##
from sklearn.svm import SVC
clf = SVC(kernel = "rbf")
print "\n\nSVM"
test_classifier(clf, my_dataset, features_list)

### deploying Naive Bayes Classifier on data and calling test_classifier function##
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
print "\n\nGaussian NB"
test_classifier(clf, my_dataset, features_list)

## implement k neighbors classfier on the data  ##
## knn on unscaled data ##
from sklearn import neighbors
clf= neighbors.KNeighborsClassifier()
print "knn"
test_classifier(clf, my_dataset, features_list)
## Tuning Adaboost Classifier
### calling adaboost classifier##
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
boost=AdaBoostClassifier()

### specifying parameters for gridsearch###
parameters={
           'n_estimators':[10,20,30,50],
           'learning_rate':[0.5,1],
            'algorithm' : ('SAMME', 'SAMME.R')
           }
#### cross validation is deployed to evaluate the model ####
cv = StratifiedShuffleSplit(labels, 100, random_state = 42)
### gridsearch searches for different combinations which give best model##
m=GridSearchCV(boost, parameters, cv=cv, scoring='f1')

### gridsearch fits various models on data to get best fit###
m.fit(features, labels)
print m.best_params_
clf=m.best_estimator_

### fits the best model on the data ### 
test_classifier(clf, my_dataset, features_list)
### scaling all features and running code##
### importing Pipeline and MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
### calling k nearest neighbors ###
knn= neighbors.KNeighborsClassifier()

### specifying parameters for knn###
parameters={'knn__n_neighbors':[5,10,15],
            'knn__weights':('uniform','distance'),
            'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            
    
}

### creating pipeline to direct flow, first the features are scaled using minmax scaler and then knn is applied to the data###
flow= Pipeline(steps=[('scaler',MinMaxScaler()),('knn',knn)])
### gridsearch function finds best parameters for knn in second step of the flow###
grid=GridSearchCV(flow, parameters, cv=cv, scoring='f1')

### different models fit on data###
grid.fit(features,labels)
### prints best fit parameters###
print grid.best_params_
### finds best model and assigns to clf###
clf=grid.best_estimator_

### best model is fit on data and evaluation metrics are observed
test_classifier(clf, my_dataset, features_list)
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

##### TUNING DECISION TREE CLASSIFIER ######
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV

data = featureFormat(my_dataset, features_list, sort_keys = True)  
labels, features = targetFeatureSplit(data)

tree = DecisionTreeClassifier()

#### specifying parameters for the gridsearch function to make different combinations###
parameters = {'min_samples_split':[5,10,15,20],
                'max_depth':[3,5,7,10],
                'max_leaf_nodes':[5,10,15],
             'criterion':('gini','entropy')}

#### cross validation is deployed to evaluate the model ####
cv = StratifiedShuffleSplit(labels, 100, random_state = 42)
### gridsearch searches for different combinations which give best model##
m=GridSearchCV(tree, parameters, cv=cv, scoring='f1')

###gridsearch fits various models on data to get best fit######
m.fit(features, labels)

### prints best parameters ###
print m.best_params_
### finds best model###
clf=m.best_estimator_

### fits the best model on the data ### 
test_classifier(clf, my_dataset, features_list)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
### classifier without new features##
##### TUNING DECISION TREE CLASSIFIER ######
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from tester import test_classifier


### finalisisng features ####
features_list=['poi','exercised_stock_options','restricted_stock','total_payments','bonus','from_poi_to_this_person',
               'shared_receipt_with_poi','other','salary','expenses']
data = featureFormat(my_dataset, features_list, sort_keys = True)  
labels, features = targetFeatureSplit(data)

tree = DecisionTreeClassifier()

#### specifying parameters for the gridsearch function to make different combinations###
parameters = {'min_samples_split':[5,10,15,20],
                'max_depth':[3,5,7,10],
                'max_leaf_nodes':[5,10,15],
             'criterion':('gini','entropy')}

#### cross validation is deployed to evaluate the model ####
cv = StratifiedShuffleSplit(labels, 100, random_state = 42)
### gridsearch searches for different combinations which give best model##
m=GridSearchCV(tree, parameters, cv=cv, scoring='f1')

###gridsearch fits various models on data to get best fit######
m.fit(features, labels)

### prints best parameters ###
print m.best_params_
### finds best model###
clf=m.best_estimator_

### fits the best model on the data ### 
test_classifier(clf, my_dataset, features_list)

