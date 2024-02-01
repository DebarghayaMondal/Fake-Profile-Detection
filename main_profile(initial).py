import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# Load the training dataset
profile_train = pd.read_csv(r'C:\Users\bitut\Desktop\profile\train.csv')

# Load the testing data
profile_test = pd.read_csv(r'C:\Users\bitut\Desktop\profile\test.csv')

# VISUALISING THE DATA

plt.figure(figsize = (20, 10))
sns.distplot(profile_train['nums/length username'])
plt.show()
# Correlation plot
plt.figure(figsize=(20, 20))
cm = profile_train.corr()
ax = plt.subplot()
sns.heatmap(cm, annot = True, ax = ax)
plt.show()

# Training and testing dataset (Inputs)
X_train = profile_train.drop(columns=['fake'])
X_test = profile_test.drop(columns=['fake'])

# Training and testing dataset (Outputs)
y_train = profile_train['fake']
y_test = profile_test['fake']

# Scale the data before training the model
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

# Define the model
model = keras.Sequential([
    Dense(64, input_dim=11, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.6),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, validation_split=0.1, callbacks=[early_stopping])

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred_classes))

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# # manual input through csv file
# manual_csv = pd.read_csv(r'C:\Users\bitut\Desktop\profile\generated_data.csv')
# manual_csv_array=np.array(manual_csv)
# manual_input_scaled = scaler_x.transform(manual_csv_array)

# # Make predictions on the manual input csv data
# manual_predictions = model.predict(manual_input_scaled)
# manual_predictions_classes = np.argmax(manual_predictions, axis=1)

# # Print the manual input predictions on the csv file
# choice=int(input("Read csv file?"))
# if(choice==1):
#     print("Manual Input CSV Predictions:")
#     for i, prediction_class in enumerate(manual_predictions_classes):
#         if prediction_class == 0:
#             print(f"Sample {i + 1}: Not Fake")
#         else:
#             print(f"Sample {i + 1}: Fake")
