from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

digits = load_digits()

x, y = digits.data, digits.target
print(y)
y = label_binarize(y, classes=list(range(10)))
print('------------------------------')
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y)
model = OneVsRestClassifier(svm.SVC(kernel='linear'))
clf = model.fit(x_train, y_train)
print(clf.score(x_train, y_train))
