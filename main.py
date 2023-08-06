import numpy as np
import sklearn.model_selection
import tensorflow
import pandas as pd

print("Hello World!")
data = pd.read_csv('student-mat.csv', sep=';')

data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]
predict = "G3"

X = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

X_train, y_train, X_test, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
