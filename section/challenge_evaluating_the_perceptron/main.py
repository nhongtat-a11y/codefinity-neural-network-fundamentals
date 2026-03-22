import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
os.system('wget https://codefinity-content-media.s3.eu-west-1.amazonaws.com/f9fc718f-c98b-470d-ba78-d84ef16ba45f/section_2/perceptron.py 2>/dev/null')
from perceptron import X, y, model

# 1. Split the dataset (80% for training set and 20% for test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
# 2. Train the model for 10 epochs with a learning rate of 0.01
model.fit(X_train, y_train, epochs=10, learning_rate = 0.01)
# 3. Obtain predictions for all examples in the test set
predictions = np.array([model.forward(test_input)[0] for test_input in X_test])
# Rounding predictions (e.g., 0.83 rounds to 1, and 0.1 to 0)
predicted_labels = np.round(predictions)
# 4. Calculate the accuracy on the test set
print(accuracy_score(y_test, predicted_labels))