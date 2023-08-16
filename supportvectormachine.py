import sklearn
from sklearn import datasets
from sklearn import svm

ds = datasets.load_breast_cancer()

# print(ds.feature_names)
# print(ds.target_names)

X = ds.data
y = ds.target


X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
classes = ds.target_names

print(classes)

