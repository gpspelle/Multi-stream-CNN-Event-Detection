
from sklearn.externals import joblib
clf = joblib.load('svm.pkl')

for i in range(0, 1000):
    ans = clf.predict(i/1000)

    if ans == 1:
        print(i/1000)
        break
