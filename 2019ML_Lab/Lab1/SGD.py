import numpy as np
import sklearn.datasets
import sklearn.model_selection
import matplotlib.pyplot as plt
from scipy.sparse import linalg
import random

# load dataset
data = sklearn.datasets.load_svmlight_file('dataset/housing_scale', n_features=13)

# split the dataset into traning set and validation set(80% for training set , 20% for validation set)
X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(data[0], data[1], test_size=0.2, random_state=222)


# loss function (using least square loss)
def Loss(y, y_):
    loss = 0.5*((y-y_)**2)
    return loss


# initialize parameter w
w = np.random.normal(size=13)


# y = wx
predict_init = X_train.dot(w)
loss_init = Loss(predict_init, Y_train)
print('initial mean loss is:{}'.format(loss_init.mean()))

w_k = linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(Y_train)
print(w_k)

# some parameters
EPOCHS = 500
LR = 0.0008 # learning rate

# initialize
loss_train = []
loss_val = []

L2_norm = []

# SGD(Stochastic Gradient Descent)

for epoch in range(EPOCHS):
    for i in range(X_train.shape[0]):
        # pick a sample randomly
        randnumber = random.randint(0, X_train.shape[0]-1)
        X = X_train[randnumber]
        Y = Y_train[randnumber]
        # gradient
        G = X.T.dot(X.dot(w)-Y)
        D = -G
        w += LR*D
    L2_norm.append(np.linalg.norm(w - w_k, ord=2))

    loss_train.append(Loss(X_train.dot(w), Y_train).mean())
    loss_val.append(Loss(X_val.dot(w), Y_val).mean())


'''
# GD
for epoch in range(EPOCHS):
    G = X_train.T.dot(X_train.dot(w)-Y_train)
    D = -G
    w += LR*D
    loss_train.append(Loss(X_train.dot(w), Y_train).mean())
    loss_val.append(Loss(X_val.dot(w), Y_val).mean())
    L2_norm.append(np.linalg.norm(w-w_k, ord=2))
'''
print('mean loss_train is:{}'.format(loss_train[-1]))
print('mean loss_val is:{}'.format(loss_val[-1]))


# plot img1
plt.figure(figsize=[15, 6])
plt.title('L2 norm optimization')
plt.xlabel('epoch')
plt.ylabel('||W_k - W*||2')
plt.plot(L2_norm, color='red', linewidth=1, label='L2 norm')
plt.legend()
plt.savefig('optimize')
plt.show()

# plot img2
plt.figure(figsize=[15, 4])
plt.title('Validation Set Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(loss_val, color='red', linewidth=1, label='valid')
plt.plot(loss_train, color='blue', linewidth=1, label='train')
plt.legend()
plt.savefig('SGD_Validation-Set-Loss.png')
plt.show()
