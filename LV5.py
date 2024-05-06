import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from mlxtend.plotting import plot_decision_regions

#zadatak 1

X, y = make_classification(n_samples = 200, n_features = 2, n_redundant = 0, n_informative = 2, random_state = 213, n_clusters_per_class = 1, class_sep = 1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

X1_train = X_train[:, 0]
X2_train = X_train[:, 1]
plt.scatter(X1_train, X2_train, c = y_train, cmap = 'magma', label = 'Train data')
plt.scatter(X_test[:, 0], X_test[:, 1], c = y_test, cmap = 'viridis', marker = 'X', label = 'Test data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()

LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)

coef = LogRegression_model.coef_.T
intercept = LogRegression_model.intercept_[0]
print(f'Coefficient: {coef}, intercept: {intercept}')
plot_decision_regions(X_train, y_train, LogRegression_model)
plt.scatter(X1_train, X2_train, c = y_train, cmap = 'Blues')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

y_test_p = LogRegression_model.predict(X_test)
cm = confusion_matrix(y_test, y_test_p)
print('Matrica zabune: ', cm)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_p))
disp.plot()
plt.show()

print('Accuracy: ', accuracy_score(y_test, y_test_p))
print('Precision: ', precision_score(y_test, y_test_p))
print('Recall: ', recall_score(y_test, y_test_p))

correct = np.where(y_test_p == y_test)[0]
wrong = np.where(y_test_p != y_test)[0]
plt.scatter(X_test[correct, 0], X_test[correct, 1], c = 'green', label = 'Correct classification')
plt.scatter(X_test[wrong, 0], X_test[wrong, 1], c = 'black', label = 'Incorrect classification')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()


#zadatak 2

labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor = 'w',
                    label=labels[cl])
    
# ucitaj podatke
df = pd.read_csv('LV5/penguins.csv')

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

train_classes, train_counts = np.unique(y_train, return_counts = True)
plt.bar(train_classes, train_counts, tick_label = ['Adelie', 'Chinstrap', 'Gentoo'])
plt.title('Number of examples per class (train)')
plt.ylabel('Number of examples')
plt.show()

test_classes, test_counts = np.unique(y_test, return_counts = True)
plt.bar(test_classes, test_counts, tick_label = ['Adelie', 'Chinstrap', 'Gentoo'])
plt.title('Number of examples per class (test)')
plt.ylabel('Number of examples')
plt.show()

LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)

print('Parameters: ', LogRegression_model.coef_)
#Razlika u odnosu na binarni klasifikacijski problem iz prvog zadatka je u broju klasa. U prvom zadatku imamo samo dvije klase (binarni problem), dok u ovom zadatku imamo tri klase.

#plot_decision_regions(X_train, y_train, classifier = LogRegression_model)
#funkcija ne radi i baca grešku kod pozivanja

y_test_p = LogRegression_model.predict(X_test)
cm = confusion_matrix(y_test, y_test_p)
print('Matrica zabune: ', cm)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_p))
disp.plot()
plt.show()

print('Accuracy: ', accuracy_score(y_test, y_test_p))
print('Classification report: ', classification_report(y_test, y_test_p))
