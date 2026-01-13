from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import pandas as pd
import os

df = pd.read_csv('diabetes.csv')

X = df.iloc[:,0:8]
y = df.iloc[:, 8]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

modelpath = "./best_model.keras"

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)

model = Sequential()
model.add(Dense(128, input_dim=8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=500, batch_size=10, validation_split=0.25, 
                    callbacks=[early_stopping_callback, checkpointer])

test = model.evaluate(X_test, y_test)
print("Test Accuracy: ", test[1])