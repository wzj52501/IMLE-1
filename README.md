# IMLE
(Updating)

## Environment

* TensorFlow 1.12.0
* Python 3.6.0

## Data preparing
* download train data(random faces data) [https://pan.baidu.com/s/10vLpSFAFHdNHDUFNPloykw]

## Training
```
python main.py -ac train
```

## Eval
```
python main.py -ac test 
```
(but the result is not very good..) \
![face1](eval/1300_faceimage65731.jpg)

## Tensorboard
```
tensorboard --logdir logs 
```
