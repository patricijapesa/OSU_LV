#import bibilioteka
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np

#učitavanje dataseta
iris = datasets.load_iris()

##################################################
#1. zadatak
##################################################


#a)
data = iris.data
names = iris.target
classes = iris.target_names

virginica = data[names == 2]
versicolour = data[names == 1]
setosa = data[names == 0]

virginica_sepal_lenght = virginica[:,0]
virginica_petal_lenght = virginica[:,2]
setosa_sepal_lenght = setosa[:,0]
setosa_petal_lenght = setosa[:,2]

plt.scatter(virginica_petal_lenght, virginica_sepal_lenght, color = 'green', label = 'Virginica')
plt.scatter(setosa_petal_lenght, setosa_sepal_lenght, color = 'grey', label = 'setosa')
plt.xlabel('Duljina latica')
plt.ylabel('Duljina čašice')
plt.legend()
plt.show()

#b)
max_sepal_width = [max(data[names == i][:, 1]) for i in range(3)]

plt.bar(classes, max_sepal_width, color = 'blue')
plt.xlabel('Klase')
plt.ylabel('Najveća širina čašice')
plt.title('Najveća širina čašice za svaku klasu cvijeta')
plt.show()
#c)
average_setosa_sepal_width = setosa[:, 1].mean()
greater_than_average = sum(setosa[:, 1] > average_setosa_sepal_width)

print('Broj jediniki koje imaju veću širinu čašice od prosječne: ', greater_than_average)


##################################################
#2. zadatak
##################################################
iris = datasets.load_iris()
X = iris.data
y = iris.target 

#a)
inertia = []
for k in range(1,10):
    km = KMeans(n_clusters = k, init='random', n_init=5, random_state=0)
    km.fit(X)
    inertia.append(km.inertia_)

optimal_k = np.argmin(np.diff(inertia)) + 2
print('Optimalna velicina K dobivena lakat metodom: ', optimal_k)

#b)
plt.figure()
plt.plot(range(1, 10), inertia)
plt.title('Lakat metoda')
plt.xlabel('X')
plt.ylabel('Y')
plt.tight_layout()
plt.show()

#c)
km = KMeans(n_clusters=optimal_k, init='random', n_init=5, random_state=0)
km.fit(X)

#d)
plt.scatter(X[:, 0], X[:, 1], c = km.labels_, cmap='viridis', label = 'Clusteri')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c = 'red', marker = 'X', label = 'Centroidi')

plt.xlabel('Duljina latica')
plt.ylabel('Širina latica')
plt.title('Klasteriranje cvijeta iris algoritmom K-srednjih vrijednosti')
plt.legend()
plt.show()

#e)

##################################################
#3. zadatak
##################################################


#a)

#b)

#c)

#d)

#e)

#f)