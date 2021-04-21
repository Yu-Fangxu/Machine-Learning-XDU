import numpy as np
from sklearn.linear_model import Perceptron
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve
def data_generator(sigma, mean, number, a1, a2, a3):
    assert a1 + a2 + a3 == 1, "a1 + a2 +a3 must equals to 1"
    
    X1 = [(a1*np.random.multivariate_normal(mean[1], sigma)+a2*np.random.multivariate_normal(mean[2], sigma)
                +a3*np.random.multivariate_normal(mean[3], sigma)) for _ in range(number)]
    X2 = ([(np.random.multivariate_normal(mean[0], sigma)) for _ in range(number)])
    label = [-1 for _ in range(number)] + [1 for _ in range(number)]
    return X1 + X2, label

if __name__ == '__main__':
    number = 300
    sigma = 0.001 * np.eye(2)
    mean_list = [np.array([1,1]), np.array([0,0]), np.array([0,1]), np.array([1,0])]
    X, label = data_generator(sigma, mean_list, 300, 0.5, 0.2, 0.3)
    clf = Perceptron(fit_intercept=True,shuffle=True)
    clf.fit(X, label)
    acc = clf.score(X, label)
    print("The accuracy of classification is:", acc)
    
    # plot the points and perception
    
    X1 = [X[i][0] for i in range(number*2)]
    Y1 = [X[i][1] for i in range(number*2)]
    line_x = np.arange(0, 1,0.001)
    line_y = line_x * (-clf.coef_[0][0] / clf.coef_[0][1]) - clf.intercept_
    plt.plot(line_x,line_y)
    plt.scatter(X1, Y1, c=label)