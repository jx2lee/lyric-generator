# **lyric-generator**


* generate Farewell-lyric
* generate lyrcis through two models, `Non-Attention` and `Attention`

## Non-Attention

* using RNN encoder-decoder
* [paper](https://arxiv.org/abs/1406.1078), word-level & using Tensorflow
* model description
>vocab size = 75762<br>
>embedding size = 100<br>
>hidden units = 256<br>
>batch size = 100<br>
>epoch = 500<br>

* code description
> `utils.py` : get dictionary & train/test set<br>
> `train.py` : train model & evaluate model (loss track)<br>
> `test.py` : recall model & generate lyrics<br>
>> `config.py` : set parameters (*modify parameters here*)<br>
>> `prepo.py` : import data & preprocess data<br>
>> `model.py` : set model using tensorflow<br>
>> `generator.py` : generate lyrics<br>

## Run

    $ python utils.py
    $ python train.py
    $ python test.py
    
    
## Result


# Attention




---
2019.07.30 made by *jaejun.lee*
