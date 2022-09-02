# Imports
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv('stroke.csv')
df = pd.DataFrame(df)

df = df.drop(labels = ["work_type", "ever_married", "Residence_type"], axis = 1) # Remove these columns from the dataset since they won't necessarily help the model learn more effectively
df.gender = df.gender.map({'Male': 0.0, 'Female': 1.0})
df.smoking_status = df.smoking_status.map({'never smoked' : 0.0, 'Unknown' : 1.0, 'formerly smoked' : 2.0, 'smokes' : 3.0})
df.head()

x = []
y = []

# Add specific parts of the dataset to x and y lists
for row in range(df.shape[0]):
  rows = []
  for point in range(len(df.loc[0]) - 1): # "- 1" because we don't want to add the last column (the labels) to the inputs section
    rows.append(df.iloc[row][point])
  x.append(rows)
  y.append(df.loc[row][-1])
  
# Balance dataset (make sure there are an even representation of instances with label 0 and label 1)
smote = SMOTE()
x, y = smote.fit_resample(x, y)

# Divide the x and y values into three sets: train, test, and validation
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size = 0.5, random_state = 1)

# Get input shape
input_shape = len(x[0])

# Create Adam optimizer
opt = Adam(learning_rate = 0.001)

# Create model
model = Sequential()

# Add an initial batch norm layer so that all the values are in a reasonable range for the network to process
model.add(BatchNormalization())
model.add(Dense(128, activation = 'relu', input_shape = [input_shape])) # Input layer
model.add(Dropout(0.4))

# Hidden layers
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.4))

model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.4))

# Output layer
model.add(Dense(1, activation = 'sigmoid')) # Sigmoid because of binary classification

# Compile model
model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
early_stopping = EarlyStopping(min_delta = 0.001, patience = 10, restore_best_weights = True)

# Train model and store training history
epochs = 200
history = model.fit(x_train, y_train, epochs = epochs, validation_data = (x_val, y_val)) # To add callbacks add 'callbacks = [early_stopping]'

# Visualize loss and validation loss
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']

plt.plot(loss, label = 'Loss')
plt.plot(val_loss, label = 'Validation Loss')
plt.title('Validation and Training Loss Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Visualize accuracy and validation accuracy
accuracy = history_dict['binary_accuracy']
val_accuracy = history_dict['val_binary_accuracy']

plt.plot(accuracy, label = 'Training Accuracy')
plt.plot(val_accuracy, label =' Validation Accuracy')
plt.title('Validation and Training Accuracy Across Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Prediction vs. actual value (change the index to view a different input and output set)
index = 0

classes = ['no stroke', 'stroke']

pred_prob = model.predict([x_test[index]])[0][0]
prediction = 1 if pred_prob > 0.5 else 0
actual_class = int(y_test[index])

# Get certainty (the probability the model thinks it is correct)
if prediction == 0:
  certainty = (1 - pred_prob) * 100
else:
  certainty = pred_prob * 100

print(f"\nModel's Prediction ({certainty}% certainty): {prediction} ({classes[prediction]}) | Actual Class: {actual_class} ({classes[actual_class]})")

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose = 0) # Change verbose to 1 or 2 for more information
print(f'\nTest accuracy: {test_acc * 100}%')

# View performance metrics
predict = model.predict(x_test)
predictions = [1.0 if j > 0.5 else 0 for j in predict] # Adjust values for classification report
print('\nClassification Report:')
print(classification_report(y_test, predictions))
