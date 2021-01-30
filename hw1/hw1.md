# Regression 

<font color=gray>refrence link:[homework1],本作业提供了数据(http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML20.html)</font>

## 1. Data Preprocessing

<font color=gray>主要参考了李宏毅老师提供的数据处理，十分推荐学习一下`numpy`数据格式的转换</font>

**train.csv** 的資料為 12 個月中，每個月取 20 天，每天 24 小時的資料(每小時資料有 18 個 features)

### 1.1 Data Conversion

![圖片說明](https://drive.google.com/uc?id=1LyaqD4ojX07oe5oDzPO99l9ts5NRyArH)

![alt text](https://drive.google.com/uc?id=1wKoPuaRHoX682LMiBgIoOP4PDyNKsJLK)

### 1.2 Data Normalization 

很简单的标准化，减去均值处以方差，使各个特征的均值为零，方差为1,请注意计算的`axis`

```python
mean_x = np.mean(x, axis = 0) #18 * 9 
std_x = np.std(x, axis = 0) #18 * 9 
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #18 * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
```

## 2.Model and optimization 

- `MODEL.py`提供了不同的model，但是需在`main.py`中调整`learning_rate`

- 使用`numpy.random.random`进行初始化（李宏毅老师中使用`numpy.zeros()`）
- 使用了`early_stop`的思想，一旦`validation_loss`上升，则停止训练

### 2.1 一次项的优化

代码提供了含`momentum`和`adagrad`的优化器，可自行进行调整

####   2.1.1简单的全局优化

<div style="text-align:center;font-weight:bold">表1 简单全局优化学习率与损失表</div>

| **learning_rate** | **val_loss** | **learning_rate** | **val_loss** |
| :---------------: | :----------: | :---------------: | :----------: |
|       1e-5        |     inf      |       1e-6        | 5.67（T16）  |
| **learning_rate** | **val_loss** | **learning_rate** | **val_loss** |
|       1e-7        |  5.67(T99)   |       5e-6        |   5.66(T4)   |

从上述表格我们可以看到其最优值是接近的，学习率的调整只是改变了收敛的快慢，那么**最优损失是全局最优还是局部最优**？

考虑的方法

- 由于初始化是随机的，可以多跑几次（对于大型数据集是不行的）

- 使用momentum

#### 2.1.2 momentum的使用

<font color=gray>learning rate is 5e-6</font>

<div style="text-align:center;font-weight:bold">表2 momentum权重与损失表</div>

|   beta1   | **val_loss** |   beta1   | **val_loss** |
| :-------: | :----------: | :-------: | :----------: |
|    0.9    |   5.66(T4)   |    0.1    |   5.66(T4)   |
| **beta1** | **val_loss** | **beta1** | **val_loss** |
|    0.5    |   5.66(T4)   |     /     |      /       |

疑问：是不是这样子可以判断到了全局最优值？

#### 2.1.3 Adagrad的使用

**Optimizer**主要是两个方向，一个是上面的动量想法，一个是学习率的自适应调整。这里感觉用不上，但是还是进行了尝试

<div style="text-align:center;font-weight:bold">表3 自适应学习率权重与损失表</div>

|   beta2   | **val_loss**  |   beta2   | **val_loss**  |
| :-------: | :-----------: | :-------: | :-----------: |
|    0.9    | 0.9(T82 S60)  |    0.1    | 5.66(T80 S77) |
| **beta2** | **val_loss**  | **beta2** | **val_loss**  |
|     0     | 5.67(T82 S60) |     /     |       /       |

- 为什么进行`beta2=0`的实验，原因在于我的自适应速率公式为
  $$
  comulative\_gradient = comulative\_gradient*beta2+|gradient|*(1-beta2) \\
  \theta = \theta-\Delta\theta \\
  \Delta \theta = \frac{learning_rate}{\epsilon + \sqrt{comulative\_gradient}}
  $$
  当`beta2=0`其实就是除以本身的开方

- 在实验过程中，发现由于自适应的作用，模型在最优点附近进行震荡，S代表了相同loss开始轮次（由于采用了`early_stop`机制，因此可以判定loss在下降，只是很慢，保留了两位小数因此结果上反映不出来）

#### 2.1.4 总结

1. 在本轮的实验中可以发现由于数据较为简单，虽然`momentum`和`adagrad`没有副作用，但是也没有起到特别好的效果。但是，经过重复实验，可能是初始化的原因，有出现`validation_loss`比上述低的情况，并且两个机制还改善了结果。

### 2.2 二次项的优化

<font color=gray>这里为了让两个模型能快速的切换，引入了python中函数可变参数的方法，若不了解，烦请自行学习</font>

<div style="text-align:center;font-weight:bold">表4 二次全局优化学习率与损失表</div>

| **learning_rate** | **val_loss** | **learning_rate** | **val_loss** |
| :---------------: | :----------: | :---------------: | :----------: |
|       1e-6        |     6.41     |       1e-5        |     5.69     |
| **learning_rate** | **val_loss** | **learning_rate** | **val_loss** |
|       1e-4        |     5.99     |       1e-2        |     inf      |

- 实验做到这里，本人发现二次型的计算明显慢了很多，这里考虑到inputs是全局优化的，因此在求导过程中对输入的平方可以进行保存。（用空间换时间，本人验证是有效的）
- 同时除了轮次应该添加运行时间来更明白地反映时间
- 模型变复杂后，学习率可以相应地增加

## 3. Summary

1. 数据预处理十分重要，本人在拿到数据之后其实对数据的处理是一头雾水的，代码完全参考了李宏毅老师
2. 模型的初始化是相当重要的，这可以从多次实验中收敛到不同`loss`看出（即使使用`momentum`，也有可能是调参导问题，并不能很好解决局部最优值）
3. 数学公式要写对，否则结果可能是天差地别
4. **最后一点可能是废话**：还是要自己动手写写代码，像时间换空间、学习率的调整这些思想，本人感觉做完实验理解的会更好些

## 实验出现的问题

1.  **自适应学习率公式错误**
   开始我的梯度积累公式为平方的和，即
   $$
   comulative\_gradient = comulative\_gradient*beta2+gradient^2*(1-beta2)
   $$
   这使得模型的梯度更新就是从1开始然后越来越小，无论`beta2`如何调整，最终都收敛在`val_loss=25.08`附近

2. **`bias_independent`实验**
   我将偏置单独提出来更新，结果是收敛速度慢很多，且效果要差一些，还没找到原因（请注意其中`train.py`的文件路径）

3. 要做一些图表的可视化

   

