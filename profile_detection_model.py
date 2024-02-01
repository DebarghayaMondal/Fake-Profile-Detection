import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

# Load the training dataset
profile_train = pd.read_csv('train.csv')

# Load the testing data
profile_test = pd.read_csv('test.csv')

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

# Define the model prediction function
def predict(inputs):
    # Make predictions using your model
    inputs_scaled = scaler_x.transform(inputs)
    predictions = model.predict(inputs_scaled)
    predictions_classes = np.argmax(predictions, axis=1)

    # Return the predictions as a dictionary
    result = {
        'predictions': predictions_classes.tolist()
    }

    return result