import numpy as np
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.models import load_model

model = load_model('model.keras')
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_test_s = x_test.astype("float32") / 255
x_test_s = np.expand_dims(x_test_s, -1)

predictions = model.predict(x_test)
y_test_p = np.argmax(predictions, axis = 1)

for i in range(100):
    if(y_test[i] != y_test_p[i]):
        plt.imshow(x_test[i])
        plt.title(f'Stvarna oznaka: {y_test[i]}, predvidena oznaka: {y_test_p[i]}')
        plt.show()