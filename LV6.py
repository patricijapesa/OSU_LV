import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

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
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost logisticka: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()  

#zadatak 1

KNN_model = KNeighborsClassifier(n_neighbors = 7)
KNN_model.fit(X_train_n, y_train)
y_train_p_KNN = KNN_model.predict(X_train)
y_test_p_KNN = KNN_model.predict(X_test)

print('KNN test accuracy: ', accuracy_score(y_test, y_test_p_KNN))
print('KNN train accuracy: ', accuracy_score(y_train, y_train_p_KNN))

plot_decision_regions(X_train_n, y_train, classifier = KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost KNN: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
plt.tight_layout()
plt.show()  

#rezultati koje dobijemo s KNN modelom su puno lošiji kada je u pitanju točnost nego rezultati dobiveni logističkom regresijom
#kada je k=1 dobivamo pretjerano usklađivanje, a kada je k=100 dobivamo podusklađivanje


#zadatak 2

k_values = [i for i in range (1, 100)]
scores = []
for k in k_values:
    KNN_model = KNeighborsClassifier(n_neighbors = k)
    score = cross_val_score(KNN_model, X_train_n, y_train, cv = 5)
    scores.append(np.mean(score))

print('Najbolji k: ', k_values[np.argmax(scores)])


#zadatak 3

SVM_model = svm.SVC(kernel = 'poly', gamma = 0.9, C = 5)
SVM_model.fit(X_train_n, y_train)
y_train_p_SVM = SVM_model.predict(X_train)
y_test_p_SVM = SVM_model.predict(X_test)

print('SVM test accuracy: ', accuracy_score(y_test, y_test_p_SVM))
print('SVM train accuracy: ', accuracy_score(y_train, y_train_p_SVM))

plot_decision_regions(X_train_n, y_train, classifier = SVM_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost SVM: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_SVM))))
plt.tight_layout()
plt.show()  

#viša vrijednost C dovodi do uže granice odluke i pretjeranom usklađivanju, 
#viša vrijednost gamma dovodi do složenijih granica odluke, 
#a promjena tipa kernela mijenja oblik granice odluke


#zadatak 4

SVM_model = svm.SVC(kernel = 'rbf')
parameters = {'C':[0.1, 1, 10], 'gamma':[0.01, 0.1, 1]}
grid_search = GridSearchCV(SVM_model, parameters, cv = 5)
grid_search.fit(X_train_n, y_train)

print('Optimalne vrijednosti hiperparametara: ', grid_search.best_params_)