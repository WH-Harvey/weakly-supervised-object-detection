#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
from torch import nn
from torchvision.models import densenet121
from torchvision.models import resnet101


# In[2]:

#freeze the former layers
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# In[3]:

#
def fine_tuning(model):
    for param in list(model.parameters())[:-3]:
        param.requires_grad = False
    for param in list(model.parameters())[-3:]:
        param.requires_grad = True


# In[ ]:

#define a model
def Model():
    
    model = densenet121(pretrained=True)
    set_parameter_requires_grad(model, True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 20)
    
    return model

