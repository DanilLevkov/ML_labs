import numpy as np


# -----------------------------------------------------------
# Simple Gradient Descent function
# x - matrix of parameters
# y - goal vector, the size must match row number of x
# alpha - speed parameter
# use_alpha_descent - if true, alpha will decrease like 1/k on k step. If false alpha is constant
# max_iteration_num - maximum number of iterrations.
# weights_closeness - stop criterion, when (w[k] - w[k-1])/w[k] < weights_closeness. If it is 0, there is no criterion
# -----------------------------------------------------------
def gradient_descent(x, y, alpha = 0.01, use_alpha_descent = False, weights_closeness = 0,  max_iteration_num = 1000):
    n = x.shape[0] # number of samples
    assert n == y.shape[0] , 'y and x sizes do not match!'
    param_num = x.shape[1] + 1
    w = np.zeros(param_num) # w[0] - is a free member
    x = np.concatenate((np.ones((n, 1)), x), axis=1)
    alpha *= 2/n
    step_koeff = alpha
    for k in range(max_iteration_num):
        y_diff = np.dot(x, w) - y # includes ... + w[0]
        if use_alpha_descent:
            step_koeff = alpha / (k+1); 
        old_w = w
        for i in range(param_num):
            w[i] -= step_koeff * np.dot(y_diff.T, x[:, i])
        if weights_closeness > 0 and np.linalg.norm(old_w - w) / np.linalg.norm(old_w) < weights_closeness:
            break
    return w

# -----------------------------------------------------------
# Stochastic Gradient Descent function
# x - matrix of parameters
# y - goal vector, the size must match row number of x
# alpha - speed parameter
# max_iteration_num - maximum number of iterrations.
# -----------------------------------------------------------
def gradient_descent_rand(x, y, alpha = 0.01, max_iteration_num = 10000):
    n = x.shape[0] # number of samples
    assert n == y.shape[0] , 'y and x sizes do not match!'
    param_num = x.shape[1] + 1
    w = np.zeros(param_num) # w[0] - is a free member
    x = np.concatenate((np.ones((n, 1)), x), axis=1)
    alpha *= 2
    for k in range(max_iteration_num):
        i = np.random.randint(n)
        y_diff = np.dot(x[i,:], w) - y[i] # now it is a number
        w -= alpha * y_diff * x[i,:].T
    return w

# -----------------------------------------------------------
# Mini Batch Gradient Descent function
# x - matrix of parameters
# y - goal vector, the size must match row number of x
# batch_part - percent of samples which are taken into account in each step
# alpha - speed parameter
# max_iteration_num - maximum number of iterrations.
# -----------------------------------------------------------
def gradient_descent_mini_batch(x, y, batch_part = 0.2, alpha = 0.01, max_iteration_num = 10000):
    n = x.shape[0] # number of samples
    assert n == y.shape[0] , 'y and x sizes do not match!'
    assert batch_part <= 1
    param_num = x.shape[1] + 1
    w = np.zeros((param_num, 1)) # w[0] - is a free member
    x = np.concatenate((np.ones((n, 1)), x), axis=1)
    samples_per_step = int(n * batch_part) + 1
    alpha *= 2/samples_per_step
    for k in range(max_iteration_num):
        batch_start = np.random.randint(n-samples_per_step)
        x_batch = x[batch_start:batch_start+samples_per_step]
        y_batch = y[batch_start:batch_start+samples_per_step]
        w = w - alpha * np.dot(x_batch.T, (np.dot(x_batch, w) - y_batch))
    return w