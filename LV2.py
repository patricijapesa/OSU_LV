import numpy as np
import matplotlib.pyplot as plt

#zadatak 1
x = np.array([1, 2, 3, 3, 1], float)
y = np.array([1, 2, 2, 1, 1], float)

plt.plot(x, y, 'r', linewidth = 1.5, marker = "*", markersize = 6)
plt.axis([0.0, 4.0, 0.0, 4.0])
plt.xlabel('x os')
plt.ylabel('y os')
plt.title('Primjer slike')
plt.show()

#zadatak 2
data = np.loadtxt("data.csv", delimiter=',', skiprows=1)

rows, cols = np.shape(data)
print(str(rows)," people were measured.")

height = data[:,1]
weight = data[:,2]

plt.scatter(height, weight, s=1)
 
plt.xlabel("Height [cm]")
plt.ylabel("Weight [kg]")
plt.title("Height to weight ratio")
plt.show()

height50 = height[::50]
weight50 = weight[::50]

plt.scatter(height50, weight50,s=5)
plt.xlabel("Height [cm]")
plt.ylabel("Weight [kg]")
plt.title("Height to weight ratio of every 50th person")
plt.show()

print("Min height: ", str(np.min(height)))
print("Max height: ", str(np.max(height)))
print("Average height: ", str(np.mean(height)))

men=data[np.where(data[:,0]==1)]
women=data[np.where(data[:,0]==0)]

print("Min male height: ", str(np.min(men[:,1])))
print("Max male height: ", str(np.max(men[:,1])))
print("Average male height: ", str(np.mean(men[:,1])))

print("Min female height: ", str(np.min(women[:,1])))
print("Max female height: ", str(np.max(women[:,1])))
print("Average female height: ", str(np.mean(women[:,1])))

#zadatak 3
img = plt.imread("road.jpg")
brightness = 50
brightened_image = np.clip(img.astype(np.uint16) + brightness, 0, 255).astype(np.uint8)
plt.figure()
plt.imshow(brightened_image)
plt.show()

width = img.shape[1]
second_quarter = width // 2
second_quarter_img = img[:, second_quarter:]
plt.figure()
plt.imshow(second_quarter_img)
plt.show()

img2 = cv2.imread("road.jpg")
rotated_img = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
plt.figure()
plt.imshow(rotated_img)
plt.show()

mirrored_img = np.flip(img, axis = 1)
plt.figure()
plt.imshow(mirrored_img)
plt.show()

#zadatak 4
black = np.zeros((50, 50), dtype=np.uint8) 
white = np.ones((50, 50), dtype=np.uint8) * 255

image = np.vstack((np.hstack((black, white)), np.hstack((white, black))))
plt.figure()
plt.imshow(image)
plt.show()