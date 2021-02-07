import numpy as np
import os.path as osp
np.random.seed(3)

x_train_path = 'dataset/X_train'
y_train_path = 'dataset/Y_train'
save_path = 'dataset'

def npy_gen():
    x_train = np.load(osp.join(save_path,'x_train.npy'))
    y_train = np.load(osp.join(save_path,'y_train.npy'))
    x_val = np.load(osp.join(save_path,'x_val.npy'))
    y_val = np.load(osp.join(save_path,'y_val.npy'))
    y_train = y_train[:,np.newaxis]
    y_val = y_val[:,np.newaxis]
    return x_train, y_train, x_val, y_val

def gen(dev_ratio=0.1):
    with open(x_train_path) as f:
        next(f)
        x_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)
    with open(y_train_path) as f:
        next(f)
        y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype=float)
    x_train, x_mean, x_std = _normalize(x_train, train=True)
    return _train_dev_split(x_train, y_train, dev_ratio)


def _normalize(X, train=True, specified_column=None, X_mean=None, X_std=None):
    # This function normalizes specific columns of X.
    # The mean and standard variance of training data will be reused when processing testing data.
    #
    # Arguments:
    #     X: data to be processed
    #     train: 'True' when processing training data, 'False' for testing data
    #     specific_column: indexes of the columns that will be normalized. If 'None', all columns
    #         will be normalized.
    #     X_mean: mean value of training data, used when train = 'False'
    #     X_std: standard deviation of training data, used when train = 'False'
    # Outputs:
    #     X: normalized data
    #     X_mean: computed mean value of training data
    #     X_std: computed standard deviation of training data

    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column], 0).reshape(1, -1)
        X_std = np.std(X[:, specified_column], 0).reshape(1, -1)

    X[:, specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
    one = np.ones([X.shape[0],1])
    X = np.concatenate([X,one],axis=1)
    return X, X_mean, X_std

def _train_dev_split(X, Y, dev_ratio = 0.25):
    # This function spilts data into training set and development set.
    #(ego) to make validation more sense, add a shuffle process
    _random_shuffle(X,Y)
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]

def _random_shuffle(data,label):
    label = label[:,np.newaxis]
    dataset=np.hstack([data,label])
    np.random.shuffle(dataset)
    return dataset[:,:-1],dataset[:,-1]

if __name__ == '__main__':
    x_train, y_train, x_val, y_val = gen()
    np.save(osp.join(save_path,'x_train.npy'),x_train)
    np.save(osp.join(save_path,'dataset/y_train.npy'), y_train)
    np.save(osp.join('dataset/x_val.npy'), x_val)
    np.save(osp.join('dataset/y_val.npy'), y_val)
    print('generate data completed and saved successfully')
