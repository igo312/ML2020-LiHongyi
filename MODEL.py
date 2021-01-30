import numpy as np
import random

seed = 333
class LINEAR(object):
    def __init__(self, dim, lr_rate=0.01, init_mode=None):
        self.lr_rate=lr_rate
        self.flag = False
        self.dim = dim
        self.m_0 = 0 # momentum param
        self.grad = 0 # commulative gradient
        self._test = False
        # choose the initialization mode(default:None), in defualt it will be initilized by `random.random(0,1)`
        self.__initialize(init_mode)
    def __initialize(self,mode):
        random.seed(seed)
        if not mode:
            # Defualt initilization
            self.w = np.random.random([self.dim,1])

    def loss(self, inputs, y_true, beta1=0.2, beta2=0.9):
        y_pre = np.matmul(inputs, self.w)
        _loss = np.sqrt(np.mean(np.abs(y_pre - y_true)))
        self.flag=True
        if not self._test:
            return _loss, self.__gradient(inputs, y_pre,y_true, beta1, beta2)
        else:
            return _loss

    def step(self, grad_w):
        if not self.flag:
            raise AssertionError('Must input a new set and calculate the loss then you can execute function `step()`')
        self.w = self.w - self.lr_rate*grad_w
        self.flag = False

    # calculate the gradient
    # 计算梯度
    def __gradient(self,inputs,y_pre, y_true, beta1, beta2):
        # a simple adam
        grad_w = 2*np.matmul(inputs.transpose(),(y_pre-y_true))
        m = self.m_0*beta1 + grad_w*(1-beta1)
        self.m_0 = m
        self.grad = self.grad*beta2 + np.abs(grad_w)*(1-beta2) # it should be `abs` instead of `square`
        return [m/np.sqrt(self.grad)]

    # To decide whether to train or test
    # 决定模式：训练&测试
    def test(self):
        self._test = True
    def train(self):
        self._test = False


class QUADRATIC(object):
    def __init__(self, dim, lr_rate=0.01, init_mode=None):
        self.lr_rate=lr_rate
        self.flag = False
        self.dim = dim
        self.m1, self.m2 = 0, 0 # momentum param
        self.grad1, self.grad2 = 0, 0 # commulative gradient
        self._test = False
        self.inputs2 = None
        # choose the initialization mode(default:None), in defualt it will be initilized by `random.random(0,1)`
        self.__initialize(init_mode)
    def __initialize(self,mode):
        random.seed(seed)
        if not mode:
            # Defualt initilization
            self.w1 = np.random.random([self.dim,1])
            self.w2 = np.random.random([self.dim,1])

    def loss(self, inputs, y_true, beta1=0.2, beta2=0.9):
        # make training fast, use more memory to accelerate training
        #TODO: if you want to do this, remenber change the code in function `__gradient`
        #if isinstance(self.inputs2,type(None)):
        #    self.inputs2 = np.square(inputs)
        #if self._test:
        #    y_pre = np.matmul(inputs, self.w1)+np.matmul(np.square(inputs), self.w2)
        #else:
        #    y_pre = np.matmul(inputs, self.w1) + np.matmul(self.inputs2, self.w2)
        y_pre = np.matmul(inputs, self.w1) + np.matmul(np.square(inputs), self.w2)
        _loss = np.sqrt(np.mean(np.square(y_pre - y_true)))
        self.flag=True
        if not self._test:
            return _loss, self.__gradient(inputs, y_pre,y_true, beta1, beta2)
        else:
            return _loss

    def step(self, *grad):
        grad_w1, grad_w2 = grad
        if not self.flag:
            raise AssertionError('Must input a new set and calculate the loss then you can execute function `step()`')
        self.w1 = self.w1 - self.lr_rate*grad_w1
        self.w2 = self.w2 - self.lr_rate*grad_w2
        self.flag = False

    # calculate the gradient
    # 计算梯度
    def __gradient(self,inputs,y_pre, y_true, beta1, beta2):
        # a simple adam
        grad_w1 = 2*np.matmul(inputs.transpose(),(y_pre-y_true))
        #grad_w2 = 2*np.matmul(self.inputs2.transpose(),(y_pre-y_true))
        grad_w2 = 2 * np.matmul(np.square(inputs).transpose(), (y_pre - y_true))
        m1, m2 = self.m1*beta1 + grad_w1*(1-beta1), self.m2*beta1 + grad_w2*(1-beta1)
        self.m1, self.m2 = m1, m2
        self.grad1, self.grad2 = self.grad1*beta2 + np.abs(grad_w1)*(1-beta2), self.grad2*beta2 + np.abs(grad_w2)*(1-beta2)
        return [m1/np.sqrt(self.grad1), m2/np.sqrt(self.grad2)]

    # To decide whether to train or test
    # 决定模式：训练&测试
    def test(self):
        self._test = True
    def train(self):
        self._test = False