# Tensorflow Predict Sentences on Farewell Lyric

**Models**:  

This repo is for Generating-Farewell-lyric on 'Neural Network' methods. I consists for 2 different modesl:  
* Non-Attention (Using [RNN encoder-decoder](https://arxiv.org/abs/1406.1078))
* Attention (Using [Attention-Based Recurrent Neural Network Models](https://arxiv.org/abs/1609.01454))

## Non-Attention

### Usage

**Data**:  

input data *(measures.txt)* is stored inside the `no-attention(or attention)/tmp` directory.  

**Dataset sizes**: about 11MB. For example:  
```
하지마 제발 그말은 하지마 그러지마
마지막 그 사랑도 했던 그말 하지마
미안하단 그말 듣고 싶지 않아
그 다음말은 너무나도 잘알아
안녕이라는 말을 할 거잖아
반복
사랑이란 다짐이 이렇게 쉽게 깨지나
아침이면 다 사라져버릴 달빛이었나
마지막이라 믿었던 사랑이 또 떠나나
다시는 만나지 못하는건가 내 사랑아
...
...
```

**Preperation**:  

Before training model, run `utils.py` to preprocess dataset. For example:  
```bash
$ python utils.py
```  
or,  
```bash
$ ./uitls.py
```  

And set **Hyperparameter** on `no-attention{attention}/core/var.py` to training model. The following is an example of `var.py`:  
```
# declare parameters
VOCAB_SIZE = 75762
EMBEDDING_SIZE = 100
HIDDENUNITS = 256
BATCH_SIZE = 100
EPOCHS = 300
KEEP_PROB = 0.5
PKL_PATH = 'tmp/'
CHECKPOINTS_PATH = 'res/checkpoints/'
```  

> * **VOCAB_SIZE** : Dictionary size
> * **EMBEDDING_SIZE** : word embedding size
> * **HIDDENUNITS** : hidden layer unit size
> * **BATCH_SIZE** : model batch size
> * **EPOCHS** : model epoch
> * **KEEP_PROB** : dropout rate
> * **PKL_PATH** : path for output pickle 
> * **CHECKPOINTS_PATH** : path for model checkpoints

**Train**:  

Start training the model, run `train.py`. `first` and `next` are the execution factors. `first` is used to start learning the first time, next is used to continue learning. For example:  
```bash
$ python train.py first
$ python train.py next
```  
or,  
```bash
$ ./train.py first
$ ./train.py next
```  

**Test**: 

To test training model, run `prediction.py`. For example:  
```bash
$ python prediction.py
```  
or,  
```bash
$ ./prediction.py
```  

## Attention

Writing..

### Usage

**Data**:  
**Preperation**:  
**Train**:  
**Test**:  

---
made by *jaejun.lee*  