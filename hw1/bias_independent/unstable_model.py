import numpy as np
import random

seed = 333
class LINEAR(object):
    def __init__(self, lr_rate=0.01, init_mode=None):
        self.lr_rate=lr_rate
        self.flag = False
        self._test = False
        self.grad_w = 0
        self.grad_b = 0
        self.m_b = 0
        self.m_w = 0
        # choose the initialization mode(default:None), in defualt it will be initilized by `random.random(0,1)`
        self.__initialize(init_mode)
    def __initialize(self,mode):
        random.seed(seed)
        if not mode:
            # Defualt initilization
            self.w = np.random.random([18*9,1])
            self.b = np.random.random(1)

    def loss(self, inputs, y_true, beta1=0, beta2=0):
        y_pre = np.matmul(inputs, self.w)+self.b
        _loss = np.sqrt(np.mean(np.square(y_pre - y_true)))
        self.flag=True
        if not self._test:
            return _loss, self.__gradient(inputs, y_pre, y_true, beta1, beta2)
        else:
            return _loss

    def step(self, grad_w, grad_b):
        if not self.flag:
            raise AssertionError('Must input a new set and calculate the loss then you can execute function `step()`')
        self.w = self.w - self.lr_rate*grad_w
        self.b = self.b - self.lr_rate*grad_b
        self.flag = False

    # calculate the gradient
    def __gradient(self,inputs,y_pre, y_true, beta1, beta2):
        grad_b = np.sum(2*(y_pre-y_true))
        grad_w = 2 * np.matmul(inputs.transpose(), (y_pre - y_true))
        mw, mb = self.m_w * beta1 + grad_w * (1 - beta1), self.m_b * beta1 + grad_b * (1 - beta1)
        self.m_w, self.m_b = mw, mb

        self.grad_w = self.grad_w * beta2 + np.abs(grad_w) * (1 - beta2)  # it should be `abs` instead of `square`
        self.grad_b = self.grad_b * beta2 + np.abs(grad_b) * (1 - beta2)
        return mw / np.sqrt(self.grad_w), mb / np.sqrt(self.grad_b)


    def test(self):
        self._test = True

    def train(self):
        self._test = False
