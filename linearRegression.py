import numpy as np
import sklearn.model_selection
from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

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

with open('studentmodel.pickle', 'wb') as file:
    pickle.dump(linear, file)

file = open('studentmodel.pickle', 'rb')

linear = pickle.load(file)
print('Coefficients: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(X_test)

for i in range(len(predictions)):
    print(predictions[i], X_test[i], y_test[i])

x = 'failures'
style.use('ggplot')
pyplot.scatter(data[x], data['G3'])
pyplot.xlabel(x)
pyplot.ylabel('g3')
pyplot.show()