---
toc: true
layout: post
description: Imagenet Transfer Learning using Fast.Ai on a Kaggle Dataset
categories: [markdown]
title: Transfer Learning for Pneumonia
---


Transfer learning was stated by Andrew Ng in his NIPS 2016 tutorial to be a key driver in the success of industrial applications. It can be shown that even in the handling of medical imaging data, that transfer learning can greatly reduce the costs of training over a dataset compared to the costs of training the same model from scratch. We will load up a simple usage given by Jeremy Howard of fast.ai, of the resnet-50 architecture, trained over the imagenet dataset. This model therefore has a, if you will, a sense of what the material world looks like. The pneumonia dataset we will be using is the kaggle dataset paultimothymooney/chest-xray-pneumonia. 

```python
data.show_batch(rows=3, figsize=(7,6))
```
![]({{ site.baseurl }}/images/files.PNG)


This dataset is a collection of thousands of labeled files of  chest x-rays. 

```python
learn = cnn_learner(data, models.resnet50, metrics=error_rate)
learn.lr_find()
learn.recorder.plot()
```
![]({{ site.baseurl }}/images/lrfind.png)

Here we use a method that was originally published in the 2015 paper [Cyclical Learning Rates for Training Neural Networks](http://arxiv.org/abs/1506.01186). By simply increasing the learning rate incrementally from a small value and only stopping once the loss started decreasing, you can plot the learning rates. By doing this, one can find the optimal learning rate amongst the plot. Based on the plot, a good learning rate to pick is 0.003, which is around the middle of the negative inclined portion of the graph. 

```python
learn.fit_one_cycle(6)
```
![]({{ site.baseurl }}/images/error.png)

Implementing a method of tuning weights for the network, cycling over our data for 6 epochs, we get an error rate of 0.034159, which means our model is over 96% accurate! Not bad! Note that this was done without unfreezing the model, as the default learning rate of the fit\_one\_cycle method is right around the value desired, and gives appreciable results for this use case, while maintaining a steady lowering of the error rate through each cycling of the data.

This model can be further improved by unfreezing the model so that the entire model is trainable instead of just the last layer. Then we can find the optimal learning rate for our trained model.
```python
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
```
![]({{ site.baseurl }}/images/lrfinder2.png)

Let's run for 4 more epochs. 

```python
learn.fit_one_cycle(4, slice(1e-5, 3e-4))
```

![]({{ site.baseurl }}/images/error2.png)

After fine tuning, we have come to an error rate of 0.017079. Now that is quite accurate, nearly 99%. This shows the power of modern tools when it comes to classification problems in object recognition.

You can play around with this experimentation with the [google colab file](https://github.com/ayanrafique/FastAiFun/blob/master/Pneumonia_detection.ipynb). You must download and upload your own kaggle.json file to use the code as is, which can be aquired after making a kaggle account.
