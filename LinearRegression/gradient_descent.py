# y = wx + b
# w is slop, b is intercept
# loss function: (actual - (wx+b))**2 = (yi - (w*x + b))**2

import numpy as np

# Create dataset
x = np.random.randn(10,1) # 10*1 array
y = 2*x + np.random.randn()

# Initialize parameters
w = 0
b = 0

# Set hyper-parameters
learning_rate = 0.01


# Create descent function
def descent(x,y,learning_rate, w, b):
    dw = 0 
    db = 0 
    N = np.shape(x)[0] # x's dimension
    for xi, yi in zip(x,y):
        # (yi - (w*x + b))**2
        dw += -2*(yi - (w*xi + b))*xi
        db += -2*(yi - (w*xi + b))
    # dw = loss of estimating w
    # db = loss of esitmating b

    #update w, b
    w -= dw*learning_rate
    b -= db*learning_rate
    return w,b



for epoch in range(1000):
    w, b = descent(x,y,learning_rate, w, b)
    y_hat = w*x + b
    loss = sum((y-y_hat)**2)
    print(f'{epoch} loss is {loss}, parameter estimates are {w}, {b}')
    



