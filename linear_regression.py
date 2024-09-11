import numpy as np
import matplotlib.pyplot as plt


# modele lineaire
# creation d'une fonction de fx = x*theta ou x est une matrice
# theta est une matrice de theta0 et theta1
def model(X, theta):
    return X.dot(theta)


# la finction cout error quadratile moyenne
def cost(X, y, theta):
    m = len(y)
    return 1 / (2 * m) * np.sum((model(X, theta) - y) ** 2)


# gradients
def grad(X, y, theta):
    m = len(y)
    return 1 / m * X.T.dot(model(X, theta) - y)


# descente de gradient
def gradient_decent(X, y, theta, learning_rate, n_ite):
    for i in range(0, n_ite):
        theta = theta - learning_rate * grad(X, y, theta)
    return theta


def train(Xkm, yprice, theta):
    theta_train = gradient_decent(Xkm, yprice, theta, 0.001, 100000)
    return theta_train


def display_prediction(xkm, yprice, Xkm, theta):
    data = model(Xkm, theta)
    data = (data * (np.max(yprice) - np.min(yprice))) + np.min(yprice)
    print(data)
    plt.title("Prix en fonction du kilometrage")
    plt.xlabel("Km parcourus")
    plt.ylabel("Prix")
    plt.scatter(xkm, yprice)
    plt.plot(xkm, data, c="r")
    plt.show()


def predict_value(theta, x, y):
    value = input("give the mileage of your car\n")
    value = int(value)
    value = (value - np.min(x)) / (np.max(x) - np.min(x))
    val = np.array([[value, 1]])
    price = model(val, theta)
    price = (price * (np.max(y) - np.min(y))) + np.min(y)
    prediction = price[0, 0]
    print("The predicted price is {}".format(prediction))
