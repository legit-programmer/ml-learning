import numpy as np
import sklearn.model_selection
from sklearn import linear_model
import pandas as pd

print("Hello World!")
data = pd.read_csv('student-mat.csv', sep=';')

data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]
predict = "G3"

X = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

linear = linear_model.LinearRegression()
linear.fit(X_train, y_train)
acc = linear.score(X_test, y_test)

print(acc)