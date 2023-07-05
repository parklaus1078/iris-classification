from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd

print("*****************************************************************************")
print("Look into Iris Data in numpy.ndarray")
# Iris Data in numpy.ndarray
iris = load_iris()              # Loading pre-saved data, iris dataset
iris_data = iris.data           # iris.data has data with features only in numpy.ndarray
iris_label = iris.target        # iris.target has label data in numpy.ndarray
print("iris_data: \n", iris_data[0:5])
print("iris_data_names: ", iris.feature_names)
print("iris_label: ", iris_label[0:5])
print("iris_label_names: ", iris.target_names)

print("\n *****************************************************************************")
print("Iris Data in ndarray to DataFrame")
# Iris Data in ndarray to DataFrame
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)  # iris_data to DataFrame
print("First 5 rows of iris_df without label: \n", iris_df.head())
iris_df['label'] = iris_label                                       # Adding label column to the iris DataFrame
print("First 5 rows of iris_df with label: \n", iris_df.head())

print("\n *****************************************************************************")
print("Separate Iris_df into Train subset and Test subset")
# Separate Iris_df into Train subset and Test subset
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=11)
# Parameters:
#   f = X = iris_data   : Variables(Features)
#   t = y = iris_label  : Results
#   test_size = 0.2     : Select 0.2 of the features as the Test subset
#   random_state = 11   : Selection seed
print('Features of Train subset, X_train: \n', X_train[0:5])
print('Features of Test subset, X_test: \n', X_test[0:5])
print('Labels of Train subset, y_train: \n', y_train[0:5])
print('Labels of Test subset, y_test: \n', y_test[0:5])

print("\n *****************************************************************************")
print("Train with the Learning Data set")
# Train with the Learning Data set
dt_clf = DecisionTreeClassifier(random_state=11)                    # Create DecisionTreeClassifier object
dt_clf.fit(X_train, y_train)                                        # fit() method is the training process

print("\n *****************************************************************************")
print("Predict results with the Test subset")
pred = dt_clf.predict(X_test)
print("Predicted outcomes: \n", pred)

print("\n *****************************************************************************")
print("Evaluating the prediction accuracy")
# Evaluation of Accuracy
accuracy = accuracy_score(y_test, pred)                             # Score the accuracy of the prediction
print('Prediction Accuracy: {0:.4f}'.format(accuracy))