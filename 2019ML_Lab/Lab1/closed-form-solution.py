import numpy as np
import sklearn.datasets
import sklearn.model_selection
from scipy.sparse import linalg,hstack,csr_matrix
import matplotlib.pyplot as plt

# load dataset in libsvm format into sparse CSR matrix
data = sklearn.datasets.load_svmlight_file('dataset/housing_scale', n_features=13)

# split the dataset into training set and validation set(80% for training set, 20% for validation set)
# data[0] means train_data,data[1] means train_target
X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(data[0], data[1], test_size=0.2, random_state=666)


# loss function(using least square loss)
def Loss(y, y_):
    Loss = 0.5*((y-y_)**2)
    return Loss


# initialize parameter w
w = np.random.normal(size=13)

# y = wx
predict_init = X_train.dot(w)
loss_init = Loss(predict_init, Y_train)
# find the initial mean loss
print('initial mean loss is:{}'.format(loss_init.mean()))

# get the closed-form solution
# cannot use numpy package (np.linalg.inv) here!!!
w = linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(Y_train)
print(w)


# training set
predict_train = X_train.dot(w)
loss_train = Loss(predict_train, Y_train)
print('mean loss_train is:{}'.format(loss_train.mean()))
# validation set
predict_val = X_val.dot(w)
loss_val = Loss(predict_val, Y_val)
print('mean loss_val is:{}'.format(loss_val.mean()))


# plot
plt.figure(figsize=[15, 7])
plt.title('Closed-form prediction')
plt.xlabel('House ID')
plt.ylabel('House Price')
plt.plot(Y_val, marker='o', color='blue', label='validation')
plt.plot(predict_val, marker='o', color='red', label='prediction')
plt.legend()
plt.savefig('prediction-pic')
plt.show()
