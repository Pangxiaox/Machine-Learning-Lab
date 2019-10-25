import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

# load dataset
X_train, Y_train = sklearn.datasets.load_svmlight_file('dataset/a9a', n_features=123)
X_val, Y_val = sklearn.datasets.load_svmlight_file('dataset/a9a.t', n_features=123)
# print(X_train.shape)
# print(Y_train.shape)

# make the row vectors into column vectors
Y_train = Y_train.reshape(Y_train.shape[0], 1)
Y_val = Y_val.reshape(Y_val.shape[0], 1)


# Hinge Loss Function
def HingeLoss(y_, y, C):
    loss = np.maximum(0, (1-y*y_))
    return C*loss.sum() / y.shape[0]

# initialize
loss_val = []
loss_train = []


# make iterations(using mini-batch gradient descent)
def svm(epochs, lr, batch_size, c):
    # initialize parameter w
    w = np.random.normal(size=(123, 1))
    for epoch in range(epochs):
        for i in range(X_train.shape[0] // batch_size):
            # randomly pick samples
            batch_index = np.random.choice(np.arange(X_train.shape[0]), batch_size)
            X = X_train[batch_index]
            Y = Y_train[batch_index]
            # print(X.shape)  //(32,123)
            # print(Y.shape)   //(32,1)

            # calculating gradient step
            x = (1 - Y*X.dot(w) < 0)
            y = Y.copy()
            y[x] = 0

            # gradient, update w
            G = w + (-1)*(X.transpose().dot(y)*c)
            # print(G.shape)
            D = -G
            w = w + lr * D

        # evaluate the loss on the validation set
        output_val = X_val.dot(w)
        output_train = X_train.dot(w)
        loss_val.append(HingeLoss(output_val, Y_val, 0.5))
        loss_train.append(HingeLoss(output_train, Y_train, 0.5))

        # mark the positive class and the negative class(SVM)
        output_val[output_val > 0] = 1
        output_val[output_val <= 0] = -1

        output_train[output_train > 0] = 1
        output_train[output_train <= 0] = -1

        print('EPOCHS:{}'.format(epoch))
        print('loss_val:{}'.format(loss_val[-1]))
        print('loss_train:{}'.format(loss_train[-1]))

    plot()


def plot():
    plt.figure(figsize=[9, 5])
    plt.title('SVM-Validation Set Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(loss_val, color='red', linewidth=1, label='valid')
    plt.plot(loss_train, color='blue', linewidth=3, label='train', linestyle=':')
    plt.legend()
    plt.savefig('Validation-Set-Loss')
    plt.show()


svm(200, 0.0003, 32, 0.05)
