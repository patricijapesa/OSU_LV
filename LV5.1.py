#import bibilioteka
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras

from keras import layers
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

##################################################
#1. zadatak
##################################################

#učitavanje dataseta
data = pd.read_csv('titanic.csv')

#a)
men = data[data['Sex'] == 'male']
women = data[data['Sex'] == 'female']

print('Broj zena u skupu podataka: ', len(women))
#b)
dead_people = data[data['Survived'] == 0]
percent_of_dead_people = len(dead_people)/len(data)

print(f'Postotak osoba koje nisu prezivjele potonuce: {round(percent_of_dead_people*100, 2)}%')
#c)
survived_men = men[men['Survived'] == 1]
survived_women = women[women['Survived'] == 1]
percent_of_survived_men = len(survived_men)/len(men)
percent_of_survived_women = len(survived_women)/len(women)
categories = ['Muskarci', 'Zene']
percentages = [percent_of_survived_men, percent_of_survived_women]
colors = ['green', 'yellow']

plt.bar(categories, percentages, color = colors)
plt.title('Postotak prezivjelih po spolu')
plt.xlabel('Spol')
plt.ylabel('Postotak prezivjelih')
plt.show()
#d)
print('Prosjecna dob prezivjelih zena: ', round(survived_women['Age'].mean(), 2))
print('Prosjecna dob prezivjelih muskaraca: ', round(survived_men['Age'].mean(), 2))

#e)
class1_men = survived_men[survived_men['Pclass'] == 1]
class2_men = survived_men[survived_men['Pclass'] == 2]
class3_men = survived_men[survived_men['Pclass'] == 3]

print('Najstariji prezivjeli muskarac u klasi 1: ', int(class1_men['Age'].max()),'\nnajstariji prezivjeli muskarac u klasi 2: ', int(class2_men['Age'].max()), '\nnajstariji prezivjeli muskarac u klasi 3: ', int(class3_men['Age'].max()))

##################################################
#2. zadatak
##################################################

#učitavanje dataseta
data = pd.read_csv('titanic.csv')

#train test split
data.dropna(axis = 0)
data.drop_duplicates()
data.reset_index(drop = True)

X = data[['Pclass', 'Sex', 'Fare', 'Embarked']]
X = pd.get_dummies(X, columns = ['Sex', 'Embarked'])
y = data['Survived'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s = sc.transform(X_test)

#a)
KNN_model = KNeighborsClassifier(n_neighbors = 5)
KNN_model.fit(X_train_s, y_train)

y_train_p = KNN_model.predict(X_train_s)
y_test_p = KNN_model.predict(X_test_s)

#b)
print('Tocnost na skupu podataka za ucenje: ', accuracy_score(y_train, y_train_p))
print('Tocnost na skupu podataka za testiranje: ', accuracy_score(y_test, y_test_p))

#c)
param_grid = {'n_neighbors': range(1, 30)}
KNN_model2 = KNeighborsClassifier()
grid_search = GridSearchCV(KNN_model2, param_grid, cv = 5)
grid_search.fit(X_train_s, y_train)
print('Najbolji hiperparametar K: ', grid_search.best_params_)
#d)
KNN_model_optimalni = KNeighborsClassifier(n_neighbors = 22)
KNN_model_optimalni.fit(X_train_s, y_train)

y_train_p_optimalni = KNN_model_optimalni.predict(X_train_s)
y_test_p_optimalni = KNN_model_optimalni.predict(X_test_s)
print('Tocnost na skupu podataka za ucenje: ', accuracy_score(y_train, y_train_p_optimalni))
print('Tocnost na skupu podataka za testiranje: ', accuracy_score(y_test, y_test_p_optimalni))
##################################################
#3. zadatak
##################################################

#učitavanje podataka:
data = pd.read_csv('titanic.csv')

data.dropna(axis = 0)
data.drop_duplicates()
data.reset_index(drop = True)

X = data[['Pclass', 'Sex', 'Fare', 'Embarked']]
X = pd.get_dummies(X, columns = ['Sex', 'Embarked'])
y = data['Survived'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s = sc.transform(X_test)

#a)
model = keras.Sequential()
model.add(layers.Input(shape = (7, )))
model.add(layers.Dense(12, activation = 'relu'))
model.add(layers.Dense(8, activation = 'relu'))
model.add(layers.Dense(4, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.summary()

#b)
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', ])

#c)
epochs = 100
batch_size = 5
history = model.fit(X_train_s, y_train, batch_size = batch_size, epochs = epochs, validation_split = 0.1)
predictions = model.predict(X_test)
y_test_predicted = np.argmax(predictions, axis = 1)

#d)
model.save('model.keras')

#e)
score = model.evaluate(X_test, y_test, verbose = 0)
print(score)

#f)
cm = confusion_matrix(y_test, y_test_p)
disp = ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot()
plt.show()
