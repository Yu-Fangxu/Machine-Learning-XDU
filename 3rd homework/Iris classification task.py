from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm


if __name__ == '__main__':
    inputdata = datasets.load_iris()
    x_train, x_test, y_train, y_test = \
        train_test_split(inputdata.data, inputdata.target, test_size = 0.3)
    ### define a one against rest SVM classifier ###
    clf= svm.SVC(decision_function_shape="ovr", kernel="poly")
    ### train the SVM
    clf.fit(x_train, y_train)
    print("The accuracy in training set:", clf.score(x_train, y_train))
    
    print("The accuracy in testing set:", clf.score(x_test, y_test))