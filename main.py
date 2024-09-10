# from linear_regression import train
from linear_regression import display_prediction, train, predict_value
from tools import load_csv_dataset
import numpy as np


def main():
    infinity = True
    data = load_csv_dataset("data.csv")
    theta = np.zeros((2, 1))
    x = np.array(data["km"])
    x_max = np.max(x)
    x_min = np.min(x)
    xkm = (x - x_min) / (x_max - x_min)
    xkm = xkm.reshape(len(xkm), 1)
    x = x.reshape(len(x), 1)
    y = np.array(data["price"])
    yprice = (y - np.min(y)) / (np.max(y) - np.min(y))
    yprice = yprice.reshape(len(yprice), 1)
    y = y.reshape(len(y), 1)
    Xkm = np.hstack((xkm, np.ones(xkm.shape)))
    while infinity:
        print("choose one the different options")
        print("1 : display")
        print("2 : train")
        print("3 : stop")
        print("4 : use the training program")
        user_input = input("")
        if user_input == "3":
            infinity = False
        if user_input == "2":
            theta = train(Xkm, yprice, theta)
        if user_input == "1":
            display_prediction(x, y, Xkm, theta)
        if user_input == "4":
            predict_value(theta, x, y)
    return


if __name__ == "__main__":
    main()
    # main(theta0, theta1)
