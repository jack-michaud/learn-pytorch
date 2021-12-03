# Learning Pytorch

I started off learning deep learning in Keras. Keras is a good
high level framework and I got some good mileage out of it, but
I wanted to be "closer to the metal".

Pytorch seemed like a good bet to start to learn. Most research
papers are implemented in Pytorch, and the high level API seemed
to closely mirror Keras's class API.

This repo is me learning Pytorch fundamentals with annotated 
code. 

## Convolutional Neural Network - CIFAR10 Classififier 

```python
from src.train_classifier import run_train
# Trains the model
run_train()
# Prints classification info
from src.test_classifier import net
```

