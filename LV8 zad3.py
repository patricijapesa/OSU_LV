import numpy as np
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
from keras.models import load_model
import keras.utils as image

model = load_model('model.keras')

img = image.load_img('test1.png', target_size = (28, 28), color_mode = 'grayscale')
img_array = image.img_to_array(img)

img_array = img_array.astype('float32') / 255
img_array_s = np.expand_dims(img_array, -1)

img_array_vector = img_array_s.reshape(-1, 784)

prediction = model.predict(img_array_vector)

plt.imshow(img_array, cmap='gray')
plt.title('Predvidena oznaka: ', prediction.argmax())
plt.show()