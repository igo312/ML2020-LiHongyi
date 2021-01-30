'''
# The problem of convergence
1. learning rate
2. initilization
'''
from MODEL import LINEAR
from train import data
itertime = 10000
if __name__ == '__main__':
    x_train, y_train, x_val, y_val = data()
    model = LINEAR(lr_rate=0.000005)
    for t in range(itertime):
        loss, [grad_w, grad_b] = model.loss(x_train, y_train)
        model.step(grad_w, grad_b)
        print('Training Times:{} loss is {:.2f}'.format(t, loss))
        if loss<1000:
            print('loss is smaller than 1000, break')
            break
