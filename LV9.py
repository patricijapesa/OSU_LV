import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt


# ucitaj CIFAR-10 podatkovni skup
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# prikazi 9 slika iz skupa za ucenje
plt.figure()
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.xticks([]),plt.yticks([])
    plt.imshow(X_train[i])

plt.show()


# pripremi podatke (skaliraj ih na raspon [0,1]])
X_train_n = X_train.astype('float32')/ 255.0
X_test_n = X_test.astype('float32')/ 255.0

# 1-od-K kodiranje
y_train = to_categorical(y_train, dtype ="uint8")
y_test = to_categorical(y_test, dtype ="uint8")

# CNN mreza
model = keras.Sequential()
model.add(layers.Input(shape=(32,32,3)))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# definiraj listu s funkcijama povratnog poziva
my_callbacks = [
    keras.callbacks.TensorBoard(log_dir = 'logs/cnn', update_freq = 100)
    keras.callbacks.TensorBoard(log_dir='logs/cnn_dropout', update_freq=100)
]

optimizer = keras.optimizers.Adam(learning_rate = 1.0)
model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.fit(X_train_n,
            y_train,
            epochs = 40,
            batch_size = 64,
            callbacks = my_callbacks,
            validation_split = 0.1)


score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Tocnost na testnom skupu podataka: {100.0*score[1]:.2f}')

'''
1. Zadatak
1.1. 
CNN mreža se sastoji od konvolucijskih slojeva, slojeva sažimanja, flatten sloja i potpuno povezanih slojeva.
1.2.
Na početku, točnost na skupu podataka za učenje i validacijskom skupu se povećavaju, dok se funkcije gubitka smanjuju. Međutim, na 7. epohi, točnost na validacijskom skupu smanjuje se u odnosu na prethodnu epohu, dok se funkcija gubitka na validacijskom skupu povećava.
1.3. 
Točnost na skupu podataka za testiranje iznosi 73.32%.

2. Zadatak
Nakon dodavanja dropout sloja između dva potpuno povezana sloja, točnost na skupu podataka za testiranje se povećala na 74.75%. To znači da su se performanse poboljšale dodavanjem dropout sloja. Dropout sloj smanjuje ovisnost modela o pojedinačnim značajkama, čime se smanjuje mogućnost overfittinga.

3. Zadatak
Dodavanjem ranog zaustavljanja s patience=5, proces učenja zaustavlja se na 11. epohi zato što se funkcija gubitka na validacijskom skupu počinje povećavati na 7. epohi. To znači da tijekom 5 uzastopnih epoha prosječna vrijednost funkcije gubitka na validacijskom skupu nije smanjena.

4. Zadatak
4.1. 
Kada se koristi jako mala veličina serije, proces učenja traje dulje. 
4.2. 
Korištenjem jako malog learning rate-a, točnost na testnom skupu podataka se smanjuje. 
4.3. 
Izbacivanjem određenih slojeva iz mreže, točnost na testnom skupu podataka se smanjuje.
4.4. 
Smanjenjem veličine skupa podataka za učenje za 50%, točnost na testnom skupu podataka se smanjuje.

'''