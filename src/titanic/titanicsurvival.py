'''
Created on 20 May 2018

@author: abhi
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv('/home/abhi/eclipse-welcome/TitanicSurvivor/src/data/train.csv')
test = pd.read_csv('/home/abhi/eclipse-welcome/TitanicSurvivor/src/data/test.csv')

print("Train Data...........")
print(train.head(7))
print("===========================================")
print("===========================================")
print("===========================================")
print("===========================================")
print("Test Data...........")
print(test.head(7))
print("===========================================")
print("===========================================")
print("===========================================")
print("===========================================")
print('Total number of passangers in the training data...', len(train))
print("===========================================")
print("===========================================")
print("===========================================")
print("===========================================")
print('Number of passangers in the training data who survived...', len(train[train['Survived'] == 1]))
print("===========================================")
print("===========================================")
print("===========================================")
print("===========================================")
print('% of men who survived', 100*np.mean(train['Survived'][train['Sex'] == 'male']))
print("===========================================")
print("===========================================")
print("===========================================")
print("===========================================")
print('% of women who survived', 100*np.mean(train['Survived'][train['Sex'] == 'female']))
print("===========================================")
print("===========================================")
print("===========================================")
print("===========================================")
print('% of passengers who survived in first class', 100*np.mean(train['Survived'][train['Pclass'] == 1]))
print("===========================================")
print("===========================================")
print("===========================================")
print("===========================================")
print('% of passengers who survived in third class', 100*np.mean(train['Survived'][train['Pclass'] == 3]))
print("===========================================")
print("===========================================")
print("===========================================")
print("===========================================")
print('% of children who survived', 100*np.mean(train['Survived'][train['Age'] < 18]))
print("===========================================")
print("===========================================")
print("===========================================")
print("===========================================")
print('% of adults who survived', 100*np.mean(train['Survived'][train['Age'] > 18]))
print("===========================================")
print("===========================================")
print("===========================================")
print("===========================================")
train['Sex'] = train['Sex'].apply(lambda x: 1 if x == 'male' else 0)
print("===========================================")
print("===========================================")
print("===========================================")
print("===========================================")
train['Age'] = train['Age'].fillna(np.mean(train['Age']))
print("===========================================")
print("===========================================")
print("===========================================")
print("===========================================")
train['Fare'] = train['Fare'].fillna(np.mean(train['Fare']))
print("===========================================")
print("===========================================")
print("===========================================")
print("===========================================")
train = train[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

X = train.drop('Survived', axis = 1)

y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

classifier = DecisionTreeClassifier()

classifier.fit(X_train, y_train)

print('Training accuracy...', accuracy_score(y_train, classifier.predict(X_train)))

print('Validation accuracy', accuracy_score(y_test, classifier.predict(X_test)))


