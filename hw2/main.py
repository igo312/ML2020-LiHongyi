import numpy as np
from dataset import gen, npy_gen
from model import SIGMOID
from tqdm import tqdm

itertime = 10000
beta1 = 0.2
beta2 = 0.1
if __name__ == '__main__':
    # two load data methods
    # 1. generate from file
    #x_train,y_train,x_val,y_val = gen(0.1)
    # 2. already execute dataset.py and get npy file, reload it
    x_train,y_train,x_val,y_val = npy_gen()
    model = SIGMOID(dim=510, lr_rate=0.0002)
    val_loss = float('inf')
    for t in tqdm(range(itertime)):
        model.train()
        loss, grad = model.loss(x_train, y_train, beta1=beta1, beta2=beta2)  # MSE loss
        model.step(*grad)
        # print('Training Times:{} loss is {:.2f}'.format(t, loss))
        if t % 100 == 0 and t > 0:
            model.test()
            loss,acc = model.loss(x_val, y_val)
            print('Validation Times:{} loss is {:.2f}, accuracy is {:.2f}'.format(t // 1000, loss, acc))
            # if you think it's too brutal to stop the training once validation loss increasing
            # you can save more validation loss and if model keep increasing three,four ... any times, then you can stop it.
            if val_loss > loss:
                val_loss = loss
            else:
                print('validation loss is increasing, stop training')
                break
    print('Times:{}, beta1 is {}, beta2 is {}, The final validation loss is {:.2f}'. \
          format(t // 1000, beta1, beta2, val_loss))