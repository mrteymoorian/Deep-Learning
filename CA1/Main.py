#!/usr/bin/env python
# coding: utf-8

# Import Libraries

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Import Model

# In[4]:


from Model import *


# Main

# In[23]:


def _plot(_Loss_train, _Loss_test, _acc_test, _acc_train, Number_of_Epochs):
    fig, ax = plt.subplots(1, 2, figsize=(18,6))
    
    ax[0].plot(range(1, Number_of_Epochs + 1), _Loss_train)
    ax[0].plot(range(1, Number_of_Epochs + 1), _Loss_test)
    ax[0].legend(['Train', 'Test'])
    ax[0].set_title("Loss per Epoch")
    ax[0].set(xlabel="Epoch", ylabel='Loss')

    ax[1].plot(range(1, Number_of_Epochs + 1), _acc_train)
    ax[1].plot(range(1, Number_of_Epochs + 1), _acc_test)
    ax[1].legend(['Train', 'Test'])
    ax[1].set_title("Accuracy per Epoch")
    ax[1].set(xlabel="Epoch", ylabel='Accuracy')
    plt.show()


# Q1

# In[6]:


df = pd.read_excel('Dry_Bean_Dataset.xlsx')
df = df.sample(frac=1).reset_index(drop=True)               #Shuffle Dataset
Data, Label = df.drop(["Class"], axis=1), df["Class"] 
Data = Normalization(Data)                                                           
Label = pd.factorize(df["Class"])[0]                        #Convert classes to numerical

X_test = Data[1:int(len(df) * 0.3)]
Y_test = Label[1:int(len(df) * 0.3)]
X_train = Data[int(len(df) * 0.3):]
Y_train = Label[int(len(df) * 0.3):]


# In[5]:


Number_of_Features = 16
Number_of_Categories = 7
layer_sizes = [Number_of_Features, 10, Number_of_Categories]
Number_of_Epochs = 200
Batch_size = 100
learning_rate = 0.03
alfa_momentum = 0
Gaussian_RBF = False


# Std = 0

# In[6]:


params, _Loss_train, _Loss_test, _acc_test, _acc_train, y_pred_test = model(X_train.to_numpy(), Y_train, X_test.to_numpy(),
                                                                            Y_test, layer_sizes, Number_of_Epochs,
                                                                            learning_rate, 0, alfa_momentum,
                                                                            Batch_size ,Gaussian_RBF)
_plot(_Loss_train, _Loss_test, _acc_test, _acc_train, Number_of_Epochs)
confusion_matrix(Y_test, y_pred_test)


# Std = 0.1

# In[7]:


params, _Loss_train, _Loss_test, _acc_test, _acc_train, y_pred_test = model(X_train.to_numpy(), Y_train, X_test.to_numpy(),
                                                                            Y_test, layer_sizes, Number_of_Epochs,
                                                                            learning_rate, 0.1, alfa_momentum,
                                                                            Batch_size ,Gaussian_RBF)
_plot(_Loss_train, _Loss_test, _acc_test, _acc_train, Number_of_Epochs)
confusion_matrix(Y_test, y_pred_test)


# Std = 1

# In[8]:


params, _Loss_train, _Loss_test, _acc_test, _acc_train, y_pred_test = model(X_train.to_numpy(), Y_train, X_test.to_numpy(),
                                                                            Y_test, layer_sizes, Number_of_Epochs,
                                                                            learning_rate, 1, alfa_momentum,
                                                                            Batch_size ,Gaussian_RBF)
_plot(_Loss_train, _Loss_test, _acc_test, _acc_train, Number_of_Epochs)
confusion_matrix(Y_test, y_pred_test)


# Std = 10

# In[6]:


params, _Loss_train, _Loss_test, _acc_test, _acc_train, y_pred_test = model(X_train.to_numpy(), Y_train, X_test.to_numpy(),
                                                                            Y_test, layer_sizes, Number_of_Epochs,
                                                                            learning_rate, 10, alfa_momentum,
                                                                            Batch_size ,Gaussian_RBF)
_plot(_Loss_train, _Loss_test, _acc_test, _acc_train, Number_of_Epochs)
confusion_matrix(Y_test, y_pred_test)


# Q2

# Not normalized

# In[11]:


df = pd.read_excel('Dry_Bean_Dataset.xlsx')
df = df.sample(frac=1).reset_index(drop=True)               #Shuffle Dataset
Data, Label = df.drop(["Class"], axis=1), df["Class"] 
#Data = Normalization(Data)                                                           
Label = pd.factorize(df["Class"])[0]                        #Convert classes to numerical

X_test = Data[1:int(len(df) * 0.3)]
Y_test = Label[1:int(len(df) * 0.3)]
X_train = Data[int(len(df) * 0.3):]
Y_train = Label[int(len(df) * 0.3):]


# In[12]:


Number_of_Features = 16
Number_of_Categories = 7
layer_sizes = [Number_of_Features, 10, Number_of_Categories]
Number_of_Epochs = 200
Batch_size = 100
learning_rate = 0.03
alfa_momentum = 0
Gaussian_RBF = False


# In[13]:


params, _Loss_train, _Loss_test, _acc_test, _acc_train, y_pred_test = model(X_train.to_numpy(), Y_train, X_test.to_numpy(),
                                                                            Y_test, layer_sizes, Number_of_Epochs,
                                                                            learning_rate, 0.1, alfa_momentum,
                                                                            Batch_size ,Gaussian_RBF)
_plot(_Loss_train, _Loss_test, _acc_test, _acc_train, Number_of_Epochs)
confusion_matrix(Y_test, y_pred_test)


# Q3

# In[37]:


df = pd.read_excel('Dry_Bean_Dataset.xlsx')
df = df.sample(frac=1).reset_index(drop=True)               #Shuffle Dataset
Data, Label = df.drop(["Class"], axis=1), df["Class"] 
Data = Normalization(Data)                                                           
Label = pd.factorize(df["Class"])[0]                        #Convert classes to numerical

X_test = Data[1:int(len(df) * 0.3)]
Y_test = Label[1:int(len(df) * 0.3)]
X_train = Data[int(len(df) * 0.3):]
Y_train = Label[int(len(df) * 0.3):]


# In[17]:


Number_of_Features = 16
Number_of_Categories = 7
layer_sizes = [Number_of_Features, 10, Number_of_Categories]
Number_of_Epochs = 200
learning_rate = 0.03
alfa_momentum = 0
Gaussian_RBF = False


# In[26]:


for Batch_size in [10, 100, 500, 1000, 5000]:
    params, _Loss_train, _Loss_test, _acc_test, _acc_train, y_pred_test = model(X_train.to_numpy(), Y_train, X_test.to_numpy(),
                                                                            Y_test, layer_sizes, Number_of_Epochs,
                                                                            learning_rate, 0.1, alfa_momentum,
                                                                            Batch_size ,Gaussian_RBF)
    print("Batch Size = ", Batch_size)
    _plot(_Loss_train, _Loss_test, _acc_test, _acc_train, Number_of_Epochs)


# Q4

# With 2 Hidden Layers

# In[21]:


Number_of_Features = 16
Number_of_Categories = 7
Number_of_Epochs = 200
Batch_size = 100
learning_rate = 0.03
alfa_momentum = 0
Gaussian_RBF = False
layer_sizes = [Number_of_Features, 3, 10, Number_of_Categories]
params, _Loss_train, _Loss_test, _acc_test, _acc_train, y_pred_test = model(X_train.to_numpy(), Y_train, X_test.to_numpy(),
                                                                            Y_test, layer_sizes, Number_of_Epochs,
                                                                            learning_rate, 0.1, alfa_momentum,
                                                                            Batch_size ,Gaussian_RBF)
_plot(_Loss_train, _Loss_test, _acc_test, _acc_train, Number_of_Epochs)
confusion_matrix(Y_test, y_pred_test)


# In[22]:


Number_of_Epochs = 200
learning_rate = 0.04
layer_sizes = [Number_of_Features, 7, 10, Number_of_Categories]
params, _Loss_train, _Loss_test, _acc_test, _acc_train, y_pred_test = model(X_train.to_numpy(), Y_train, X_test.to_numpy(),
                                                                            Y_test, layer_sizes, Number_of_Epochs,
                                                                            learning_rate, 0.1, alfa_momentum,
                                                                            Batch_size ,Gaussian_RBF)
_plot(_Loss_train, _Loss_test, _acc_test, _acc_train, Number_of_Epochs)
confusion_matrix(Y_test, y_pred_test)


# In[23]:


Number_of_Epochs = 200
learning_rate = 0.04
layer_sizes = [Number_of_Features, 7, 20, Number_of_Categories]
params, _Loss_train, _Loss_test, _acc_test, _acc_train, y_pred_test = model(X_train.to_numpy(), Y_train, X_test.to_numpy(),
                                                                            Y_test, layer_sizes, Number_of_Epochs,
                                                                            learning_rate, 0.1, alfa_momentum,
                                                                            Batch_size ,Gaussian_RBF)
_plot(_Loss_train, _Loss_test, _acc_test, _acc_train, Number_of_Epochs)
confusion_matrix(Y_test, y_pred_test)


# Q5

# Without Momentum

# In[38]:


Number_of_Features = 16
Number_of_Categories = 7
layer_sizes = [Number_of_Features, 10, Number_of_Categories]
Number_of_Epochs = 20
Batch_size = 100
learning_rate = 0.03
alfa_momentum = 0
Gaussian_RBF = False


# In[39]:


params, _Loss_train, _Loss_test, _acc_test, _acc_train, y_pred_test = model(X_train.to_numpy(), Y_train, X_test.to_numpy(),
                                                                            Y_test, layer_sizes, Number_of_Epochs,
                                                                            learning_rate, 0.1, alfa_momentum,
                                                                            Batch_size ,Gaussian_RBF)
_plot(_Loss_train, _Loss_test, _acc_test, _acc_train, Number_of_Epochs)
confusion_matrix(Y_test, y_pred_test)


# With momentum

# In[40]:


alfa_momentum = 0.8


# In[41]:


params, _Loss_train, _Loss_test, _acc_test, _acc_train, y_pred_test = model(X_train.to_numpy(), Y_train, X_test.to_numpy(),
                                                                            Y_test, layer_sizes, Number_of_Epochs,
                                                                            learning_rate, 0.1, alfa_momentum,
                                                                            Batch_size ,Gaussian_RBF)
_plot(_Loss_train, _Loss_test, _acc_test, _acc_train, Number_of_Epochs)
confusion_matrix(Y_test, y_pred_test)


# Q6

# In[7]:


Number_of_Features = 16
Number_of_Categories = 7
layer_sizes = [Number_of_Features, 10, Number_of_Categories]
Number_of_Epochs = 200
Batch_size = 100
learning_rate = 0.03
alfa_momentum = 0.4
Gaussian_RBF = True


# In[8]:


params, _Loss_train, _Loss_test, _acc_test, _acc_train, y_pred_test = model(X_train.to_numpy(), Y_train, X_test.to_numpy(),
                                                                            Y_test, layer_sizes, Number_of_Epochs,
                                                                            learning_rate, 0.1, alfa_momentum,
                                                                            Batch_size ,Gaussian_RBF)
_plot(_Loss_train, _Loss_test, _acc_test, _acc_train, Number_of_Epochs)
confusion_matrix(Y_test, y_pred_test)


# Q7

# In[26]:


Number_of_Features = 16
Number_of_Categories = 7
layer_sizes = [Number_of_Features, 10, Number_of_Categories]
Number_of_Epochs = 200
Batch_size = 100
learning_rate = 0.03
alfa_momentum = 0.8
Gaussian_RBF = False


# In[27]:


params, _Loss_train, _Loss_test, _acc_test, _acc_train, y_pred_test = model(X_train.to_numpy(), Y_train, X_test.to_numpy(),
                                                                            Y_test, layer_sizes, Number_of_Epochs,
                                                                            learning_rate, 0.1, alfa_momentum,
                                                                            Batch_size ,Gaussian_RBF)
_plot(_Loss_train, _Loss_test, _acc_test, _acc_train, Number_of_Epochs)
confusion_matrix(Y_test, y_pred_test)


# Q8

# In[11]:


df = pd.read_excel('Dry_Bean_Dataset.xlsx')
df1 = df[df["Class"] != "BOMBAY"]
df2 = df[df["Class"] == "BOMBAY"][0:100]
df = pd.concat([df1, df2])
df = df.sample(frac=1).reset_index(drop=True)               #Shuffle Dataset
Data, Label = df.drop(["Class"], axis=1), df["Class"] 
Data = Normalization(Data)                                                           
Label = pd.factorize(df["Class"])[0]                        #Convert classes to numerical

X_test = Data[1:int(len(df) * 0.3)]
Y_test = Label[1:int(len(df) * 0.3)]
X_train = Data[int(len(df) * 0.3):]
Y_train = Label[int(len(df) * 0.3):]


# In[12]:


Number_of_Features = 16
Number_of_Categories = 7
layer_sizes = [Number_of_Features, 10, Number_of_Categories]
Number_of_Epochs = 200
Batch_size = 100
learning_rate = 0.03
alfa_momentum = 0.8
Gaussian_RBF = False


# In[13]:


params, _Loss_train, _Loss_test, _acc_test, _acc_train, y_pred_test = model(X_train.to_numpy(), Y_train, X_test.to_numpy(),
                                                                            Y_test, layer_sizes, Number_of_Epochs,
                                                                            learning_rate, 0.1, alfa_momentum,
                                                                            Batch_size ,Gaussian_RBF)
_plot(_Loss_train, _Loss_test, _acc_test, _acc_train, Number_of_Epochs)
confusion_matrix(Y_test, y_pred_test)


# Oversampling (Repetition)

# In[20]:


df = pd.read_excel('Dry_Bean_Dataset.xlsx')
df1 = df[df["Class"] != "BOMBAY"]
df2 = df[df["Class"] == "BOMBAY"][0:100]
df = pd.concat([df1, df2, df2, df2, df2])
df = df.sample(frac=1).reset_index(drop=True)               #Shuffle Dataset
Data, Label = df.drop(["Class"], axis=1), df["Class"] 
Data = Normalization(Data)                                                           
Label = pd.factorize(df["Class"])[0]                        #Convert classes to numerical

X_test = Data[1:int(len(df) * 0.3)]
Y_test = Label[1:int(len(df) * 0.3)]
X_train = Data[int(len(df) * 0.3):]
Y_train = Label[int(len(df) * 0.3):]


# In[21]:


Number_of_Features = 16
Number_of_Categories = 7
layer_sizes = [Number_of_Features, 10, Number_of_Categories]
Number_of_Epochs = 200
Batch_size = 100
learning_rate = 0.03
alfa_momentum = 0.8
Gaussian_RBF = False


# In[22]:


params, _Loss_train, _Loss_test, _acc_test, _acc_train, y_pred_test = model(X_train.to_numpy(), Y_train, X_test.to_numpy(),
                                                                            Y_test, layer_sizes, Number_of_Epochs,
                                                                            learning_rate, 0.1, alfa_momentum,
                                                                            Batch_size ,Gaussian_RBF)
_plot(_Loss_train, _Loss_test, _acc_test, _acc_train, Number_of_Epochs)
confusion_matrix(Y_test, y_pred_test)


# In[ ]:




