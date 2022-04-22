import tensorflow as tf

#helpers
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#labels 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)

print(len(train_labels))

print(train_labels)

print(test_images.shape)

print(len(test_labels))

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# must be range 0 to 1 
train_images = train_images / 255.0
test_images = test_images / 255.0

# shows first 25 images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# builds model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), # changes from 2d array into 1d array
    tf.keras.layers.Dense(128, activation='relu'), # layer has 128 nodes (or neurons)
    tf.keras.layers.Dense(10)                      # The second (and last) layer returns a logits array with length of 10
])

# compile before training
# optimizer -  is how the model is updated based on the data it sees and its loss function.
# loss - measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
# metrics - Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.
model.compile(optimizer='adam',                                                     
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# train model
model.fit(train_images, train_labels, epochs=10)

# 
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)






