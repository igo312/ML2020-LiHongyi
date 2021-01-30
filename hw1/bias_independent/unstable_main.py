'''
# The problem of convergence
1. learning rate
2. initilization
'''
#from MODEL import LINEAR
from unstable_model import LINEAR
from train import data
import math

itertime = 100000
beta1=0.5
beta2=0.1
if __name__ == '__main__':
    x_train, y_train, x_val, y_val = data()
    #model = LINEAR(dim=18*9,lr_rate=0.000001)
    model = LINEAR(lr_rate=0.000001)
    val_loss = float('inf')
    for t in range(itertime):
        model.train()
        loss, [grad_w,grad_b] = model.loss(x_train, y_train, beta1=beta1, beta2=beta2)       # MSE loss
        model.step(grad_w,grad_b)
        # print('Training Times:{} loss is {:.2f}'.format(t, loss))
        if t%1000==0 and t>0:
            model.test()
            loss = model.loss(x_val, y_val)
            print('Validation Times:{} loss is {:.2f}'.format(t//1000, loss))
            # if you think it's too brutal to stop the training once validation loss increasing
            # you can save more validation loss and if model keep increasing three,four ... any times, then you can stop it.
            if val_loss > loss: val_loss = loss
            else:
                print('validation loss is increasing, stop training')
                break
    print('Times:{}, beta1 is {}, beta2 is {}, The final validation loss is {:.2f}'.\
          format(t//1000, beta1, beta2, val_loss))
    # beta1=0.2, beta=0.9 25.8
    #Times:41 beta1=0.2, beta=0.2 25.89