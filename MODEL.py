import numpy as np
import random

seed = 333
class LINEAR(object):
    def __init__(self, lr_rate=0.01, init_mode=None):
        # choose the initialization mode(default:None), in defualt it will be initilized by `random.random(0,1)`
        self.__initialize(init_mode)
        self.lr_rate=lr_rate
        self.flag = False
    def __initialize(self,mode):
        random.seed(seed)
        if not mode:
            # Defualt initilization
            self.w = np.random.random([18*9,1])
            self.b = np.random.random(1)

    def loss(self, inputs, y_true):
        y_pre = np.matmul(inputs, self.w)+self.b
        _loss = np.sum(np.square(y_pre - y_true))
        self.flag=True
        return _loss, self.__gradient(inputs, y_pre,y_true)

    def step(self, grad_w, grad_b):
        if not self.flag:
            raise AssertionError('Must input a new set and calculate the loss then you can execute function `step()`')
        self.w = self.w - self.lr_rate*grad_w
        self.b = self.b - self.lr_rate*grad_b
        self.flag = False

    # calculate the gradient
    def __gradient(self,inputs,y_pre, y_true):
        grad_w = 2*np.matmul(inputs.transpose(),(y_pre-y_true))
        grad_b = 2*(y_pre-y_true)
        return [grad_w, grad_b]
