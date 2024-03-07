---
author: Satyan Sharma
title: Coordinate Descent from Scratch
date: 2022-06-01
math: true
tags: ["Machine Learning"]
thumbnail: /th/th_coordesc.png
---

### Some notations:
**$n$** - number of observations, **$d$** - number of features

### Lasso cost function

$$
\large{L(w) = \frac{1}{2} \Sigma_{i=1}^{n}(y_{i} - \Sigma_{j=0}^{d}{w_{j}x_{ij}})^2 + \lambda\Sigma_{j=0}^{d}\lvert w_{j} \rvert}
$$

Taking the derivative with respect to the $w_{l}$

$$
\large{\frac{\partial L}{\partial w_{l}} = \Sigma_{i=1}^{n}(y_{i} - \Sigma_{j=0}^{d}{w_{j}x_{ij}}) \cdot x_{il} + \lambda \frac{\partial \lvert w_{l} \rvert}{\partial w_{l}}}
$$

Can be rewritten as:
$$
\large{\frac{\partial L}{\partial w_{l}} = \Sigma_{i=1}^{n}(y_{i} - \Sigma_{j \neq l}^{d}{w_{j}x_{ij}} - w_{l}x_{il}) \cdot x_{il} + \lambda \frac{\partial \lvert w_{l} \rvert}{\partial w_{l}}}
$$

$$
\large{\frac{\partial L}{\partial w_{l}} = \Sigma_{i=1}^{n}(y_{i} - \Sigma_{j \neq l}^{d}{w_{j}x_{ij}}) \cdot x_{il} +   w_{l}\Sigma_{i=1}^{n}x_{il}^2 + \lambda \frac{\partial \lvert w_{l} \rvert}{\partial w_{l}}}
$$

let 
$$
\large{a_{l} = \Sigma_{i=1}^{n}x_{il}^2}
$$
which, in case of normalized features,
$$
\large{a_{l} = \Sigma_{i=1}^{n}x_{il}^2 = 1}
$$

and let,
$$
\large{\rho_{l} = \Sigma_{i=1}^{n}(\Sigma_{j \neq l}^{d}{w_{j}x_{ij}} -y_{i}) \cdot x_{il}}
$$

with $\rho_{l}$ basically a function of the prediction without the feature $l$.

Ignoring the constant 2, we can rewrite
$$
\large{\frac{\partial L}{\partial w_{l}} = \rho_{l} + w_{l}a_{l} + \lambda \frac{\partial \lvert w_{l} \rvert}{\partial w_{l}}}
$$

### The final partial derivative equation:
<img src="/corr_Desc_01.png" alt="drawing" width="400"/>

### Soft threshold function 

The closed form solution, for the normalized features 

<img src="/corr_Desc_02.png" alt="drawing" width="400"/>

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from sklearn import linear_model
from sklearn import datasets

%matplotlib inline
plt.style.use('seaborn-white')
```

## Load Data


```python
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target.reshape(-1,1)
```


```python
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target.reshape(-1,1)
```

## Define Soft threshold


```python
def soft_threshold(rho,lamda):
    '''Soft threshold function used for normalized data and lasso regression'''
    if rho < - lamda:
        return (rho + lamda)
    elif rho >  lamda:
        return (rho - lamda)
    else: 
        return 0
    
```

## Define Coordnate descent


```python
def coordinate_descent(theta,X,y,lamda = .01, num_iters=100, intercept = False):
    '''Coordinate gradient descent for lasso regression - for normalized data. 
    The intercept parameter allows to specify whether or not we regularize theta_0'''
    
    #Initialisation of useful values 
    m,n = X.shape
    X = X / (np.linalg.norm(X,axis = 0)) #normalizing X in case it was not done before
    
    #Looping until max number of iterations
    for i in range(num_iters): 
        
        #Looping through each coordinate
        for j in range(n):
            
            #Vectorized implementation
            X_j = X[:,j].reshape(-1,1)
            y_pred = X @ theta
            rho = X_j.T @ (y - y_pred  + theta[j]*X_j)
        
            #Checking intercept parameter
            if intercept == True:  
                if j == 0: 
                    theta[j] =  rho 
                else:
                    theta[j] =  soft_threshold(rho, lamda)  

            if intercept == False:
                theta[j] =  soft_threshold(rho, lamda)   
            
    return theta.flatten()
```

## Run Lasso


```python
# Initialize variables
m,n = X.shape
initial_theta = np.ones((n,1))
theta_list = list()
lamda = np.logspace(0,4,300)/10 #Range of lambda values

#Run lasso regression for each lambda
for l in lamda:
    theta = coordinate_descent(initial_theta,X,y,lamda = l, num_iters=100)
    theta_list.append(theta)

#Stack into numpy array
theta_lasso = np.stack(theta_list).T
```

## Plot paths


```python
n,_ = theta_lasso.shape
plt.figure(figsize = (12,8))

for i in range(n):
    plt.plot(lamda, theta_lasso[i], label = diabetes.feature_names[i])

plt.xscale('log')
plt.xlabel('Log($\\lambda$)')
plt.ylabel('Coefficients')
plt.title('Lasso Paths - Numpy implementation')
plt.legend()
plt.axis('tight')
```




    (0.06309573444801933, 1584.893192461114, -849.8147108555953, 820.6104516733536)




    
![png](/coor_desc.png)
    



```python

```
### Reference:
* https://www.kaggle.com/mcweng24/lasso-regression-using-numpy
* https://xavierbourretsicotte.github.io/coordinate_descent.html
