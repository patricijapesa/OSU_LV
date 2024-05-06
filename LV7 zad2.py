import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread('imgs/imgs/test_1.jpg')

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

colors_num = np.unique(img_array)
print('Broj boja u originalnoj slici: ', len(colors_num))

km = KMeans(n_clusters = 15, init = 'random', n_init = 5, random_state = 0)
km.fit(img_array_aprox)

centers = km.cluster_centers_
labels = km.labels_

for i in range(len(img_array)):
    img_array_aprox[i] = centers[labels[i]]

img_finished = np.reshape(img_array_aprox, (w, h, d))

plt.figure()
plt.title("Kvantizirana slika")
plt.imshow(img_finished)
plt.tight_layout()
plt.show()

#povecavanjem K kvantizirana slika je sličnija originalnoj, kod primjerice K=5 razlike su očite, dok su kod K=15 manje primjetne

J_values = []
for i in range(1, 10):
    km = KMeans(n_clusters = i, init = 'random', n_init = 5, random_state = 0)
    km.fit(img_array_aprox)
    J_values.append(km.inertia_)

plt.figure()
plt.plot(range(1, 10), J_values)
plt.title('Lakat metoda')
plt.xlabel('X')
plt.ylabel('Y')
plt.tight_layout()
plt.show()

#na grafu se vidi lakat te bi rekla da je optimalna vrijednost X=2

for i in range(10):
    img_bin = np.zeros((w*h, d))
    img_bin[labels == i] = 1
    img_bin = np.reshape(img_bin, (w, h, d))
    plt.figure()
    plt.imshow(img_bin, cmap = 'gray')
    plt.title(f'Grupa {i+1}')
    plt.show()

#vec nakon 2 grupe boja jasno se vide razlike i tekst se moze procitati