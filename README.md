# CIFAR_naren

# Initialiize/define libraries:
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to range [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Analyse the data and its shape

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
          'dog', 'frog', 'horse', 'ship', 'truck']
idx = 1
plt.imshow(x_train[idx,0:])
print(labels[y_train[idx][0]])
classes_name = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
classes, counts = np.unique(y_train, return_counts=True)
plt.barh(classes_name, counts)
plt.title('Class distribution in training set')
classes_name = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
classes, counts = np.unique(y_test, return_counts=True)
plt.barh(classes_name, counts)
plt.title('Class distribution in training set')
# Convert labels to binary (1 if class is 0, 0 otherwise)
y_train_binary = np.where(y_train == 0, 1, 0)
y_test_binary = np.where(y_test == 0, 1, 0)
# Define input shape
input_shape = x_train.shape[1:]
# Build a simple neural network model
model = Sequential([
    Input(shape=input_shape),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
# BinaryCrossentropy, Categorical Cross-Entropy, SparseCategoricalCrossentropy, Mean Squared Error, Mean Absolute Error)
# Define the learning rate
learning_rate = 0.001

# Compile the model with specified learning rate
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='binary_crossentropy',
              # loss= 'categorical_crossentropy',
              # loss= 'sparse_categorical_crossentropy',
              # loss= 'mean_squared_error',
              # loss= 'mean_absolute_error',
              
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
# Evaluate the model on the testing data
y_pred_proba = model.predict(x_test)
y_pred = (y_pred_proba > 0.5).astype(np.int32)
accuracy = accuracy_score(y_test_binary, y_pred)
print("Accuracy:", accuracy)
# Print classification report
print(classification_report(y_test_binary, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test_binary, y_pred)
plt.imshow(cm, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1], ['Not Class 0', 'Class 0'])
plt.yticks([0, 1], ['Not Class 0', 'Class 0'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
# label mapping

labels = '''airplane automobile bird cat deerdog frog horseship truck'''.split()

# select the image from our test dataset
image_number = 1

# display the image
plt.imshow(x_test[image_number])

# load the image in an array
n = np.array(x_test[image_number])

# reshape it
p = n.reshape(1, 32, 32, 3)

# pass in the network for prediction and [Uploading CIFAR_10.ipynb…]()

# save the predicted label
predicted_label = labels[model.predict(p).argmax()]

# load the original label
original_label = labels[int(y_test[image_number][0])]

# display the result
print("Original label is {} and predicted label is {}".format(
	original_label, predicted_label))

