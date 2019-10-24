import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

# load dataset
X_train, Y_train = sklearn.datasets.load_svmlight_file('dataset/a9a', n_features=123)
X_val, Y_val = sklearn.datasets.load_svmlight_file('dataset/a9a.t', n_features=123)

# make the above Y_train and Y_val row vectors into column vectors
Y_train = Y_train.reshape(Y_train.shape[0], 1)
Y_val = Y_val.reshape(Y_val.shape[0], 1)

# binary classification of class0 and class1(not class -1 ) here
Y_train[Y_train == -1] = 0
Y_val[Y_val == -1] = 0


# sigmoid function
def SigmoidFunction(x):
    return 1 / (1 + np.exp(-x))


# Log-Likehood loss function
def LogLoss(y_, y):
    return -1 / y.shape[0] * (y*np.log(SigmoidFunction(y_)) + (1-y)*np.log(1-SigmoidFunction(y_))).sum()


# initialize
loss_val = []
loss_train = []

# make iterations, using mini-batch stochastic gradient descent
def logisticregression(epochs, lr, batch_size):
    # initialize parameter w
    w = np.random.normal(1, 1, size=(123, 1))
    for epoch in range(epochs):
        for i in range(X_train.shape[0] // batch_size):
            # randomly pick samples
            batch_index = np.random.choice(np.arange(X_train.shape[0]), size=batch_size)
            X = X_train[batch_index]
            Y = Y_train[batch_index]

            # gradient
            G = X.transpose().dot(SigmoidFunction(X.dot(w))-Y) / X.shape[0]
            D = -G
            w += lr*D

        # evaluate the loss on the validation set
        output_val = X_val.dot(w)
        output_train = X_train.dot(w)
        loss_val.append(LogLoss(output_val, Y_val))
        loss_train.append(LogLoss(output_train, Y_train))

        # classify the samples based on the sigmoid function
        output_val[SigmoidFunction(output_val) > 0.5] = 1
        output_val[SigmoidFunction(output_val) <= 0.5] = 0

        print('EPOCHS:{}'.format(epoch))
        print('loss_val is:{}'.format(loss_val[-1]))
        print('loss_train is:{}'.format(loss_train[-1]))

    plot()


def plot():
    plt.figure(figsize=[15, 5])
    plt.title('LR-Validation Set Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(loss_val, color='red', linewidth=1, label='valid')
    plt.plot(loss_train, color='blue', linewidth=3, label='train', linestyle=':')
    plt.legend()
    plt.savefig('LR_ValidationLoss')
    plt.show()


logisticregression(200, 0.0001, 32)
