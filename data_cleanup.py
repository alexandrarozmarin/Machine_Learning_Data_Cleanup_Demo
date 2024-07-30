#import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

#read in dataset
address = '/Users/AlexandraRozmarin/Desktop/Machine_Learning_Data_Cleanup_Demo/iris.csv'
dataset = pd.read_csv(address)

#species in dataset
dataset.Species.unique() #array(['setosa', 'versicolor', 'virginica'], dtype=object)

#separating features and labels
X = dataset.iloc[:, 1:5]
y = dataset.iloc[:,5]

#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Training Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

#Predict Species Based on Data
y_predict = clf.predict(X_test)

#Evaluation
accuracy = metrics.accuracy_score(y_test, y_predict)
print("Accuracy", accuracy) #Accuracy exceeds 95%
