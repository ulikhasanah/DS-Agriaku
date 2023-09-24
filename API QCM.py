from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics

app = Flask(__name__)

# Load and preprocess the data
df1 = pd.read_csv('/Users/jkt-ltp-038/Downloads/agriaku_20230804_qcm_dataset-master/dataset/QCM3.csv', sep=';')
df2 = pd.read_csv('/Users/jkt-ltp-038/Downloads/agriaku_20230804_qcm_dataset-master/dataset/QCM6.csv', sep=';')
df3 = pd.read_csv('/Users/jkt-ltp-038/Downloads/agriaku_20230804_qcm_dataset-master/dataset/QCM7.csv', sep=';')
df4 = pd.read_csv('/Users/jkt-ltp-038/Downloads/agriaku_20230804_qcm_dataset-master/dataset/QCM10.csv', sep=';')
df5 = pd.read_csv('/Users/jkt-ltp-038/Downloads/agriaku_20230804_qcm_dataset-master/dataset/QCM12.csv', sep=';')
data = pd.concat([df1, df2, df3, df4, df5])
X = data.iloc[:, 0:10].values
y = data.iloc[:, [10, 11, 12, 13, 14]].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Create and train the ANN model
classifier = Sequential()
classifier.add(Dense(units=10, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=7, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=5, kernel_initializer='uniform', activation='softmax'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = classifier.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=3000, epochs=8000, verbose=1)

# Define a route for plotting the loss and accuracy
@app.route('/plot_loss_accuracy', methods=['GET'])
def plot_loss_accuracy():
    f, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(history.history['loss'])
    axes[0].plot(history.history['val_loss'])
    axes[0].set_xlabel('Loss', fontsize=14)
    axes[0].set_ylabel('Epoch', fontsize=14)
    axes[0].yaxis.tick_left()
    axes[0].legend(['Train', 'Test'], loc='upper left')

    axes[1].plot(history.history['accuracy'])
    axes[1].plot(history.history['val_accuracy'])
    axes[1].set_xlabel('Accuracy', fontsize=14)
    axes[1].set_ylabel('Epoch', fontsize=14)
    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.tick_right()
    axes[1].legend(['Train', 'Test'], loc='upper left')

    plt.tight_layout()
    plt.savefig('loss_accuracy_plot.png')
    return jsonify({'message': 'Loss and accuracy plot saved as loss_accuracy_plot.png'})

# Define a route for getting the confusion matrix
@app.route('/confusion_matrix', methods=['GET'])
def confusion_matrix_route():
    # Make predictions
    y_pred_prob = classifier.predict(X_test)
    y_pred = y_pred_prob.argmax(axis=1)
    cm = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred)
    return jsonify({'confusion_matrix': cm.tolist()})

if __name__ == '__main__':
    app.run(debug=True)