import numpy as np
import random

seed = 3
class SIGMOID(object):
    def __init__(self, dim, lr_rate=0.01, init_mode=None):
        self.lr_rate=lr_rate
        self.flag = False
        self.dim = dim
        self.m_0 = 0 # momentum param
        self.grad = 0 # commulative gradient
        self._test = False
        self.epsilon = 1e-9
        # choose the initialization mode(default:None), in defualt it will be initilized by `random.random(0,1)`
        self.__initialize(init_mode)
    def __initialize(self,mode):
        random.seed(seed)
        if not mode:
            # Defualt initilization
            self.w = np.random.random([self.dim,1])

    def loss(self, inputs, y_true, beta1=0.2, beta2=0.9):
        y_pre = self._sigmoid(np.matmul(inputs, self.w))
        _loss = self._loss(y_pre, y_true)
        self.flag=True
        if not self._test:
            return _loss, self._gradient(inputs, y_pre,y_true, beta1, beta2)
        else:
            return _loss, self._accuracy(y_pre, y_true)

    def _loss(self, pre, logits):
        # cross entropy
        return np.sum(-
                      (np.multiply(logits,np.log(pre+self.epsilon))
                        +
                       np.multiply((1-logits),np.log(1-pre+self.epsilon))))

    def step(self, grad_w):
        if not self.flag:
            raise AssertionError('Must input a new set and calculate the loss then you can execute function `step()`')
        self.w = self.w - self.lr_rate*grad_w
        self.flag = False

    # calculate the gradient
    # 计算梯度
    def _gradient(self,inputs,y_pre, y_true, beta1, beta2):
        # a simple adam
        grad_w = np.matmul(inputs.transpose(),(y_pre-y_true))
        m = self.m_0*beta1 + grad_w*(1-beta1)
        self.m_0 = m
        self.grad = self.grad*beta2 + np.abs(grad_w)*(1-beta2) # it should be `abs` instead of `square`
        return [m/(np.sqrt(self.grad)+self.epsilon)]

    # To decide whether to train or test
    # 决定模式：训练&测试
    def test(self):
        self._test = True
    def train(self):
        self._test = False

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _accuracy(self, pre,logits):
        return 1-(np.where(abs(pre-logits)>0.5)[0].shape[0]/ logits.shape[0])

