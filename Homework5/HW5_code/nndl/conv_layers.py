import numpy as np
from nndl.layers import *
import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

def conv2d(im, kernel, stride=1, bias=0):
    height, width = im.shape
    kernel_size_h, kernel_size_w = kernel.shape
    out_h = int((height - kernel_size_h) / stride + 1)
    out_w = int((width - kernel_size_w) / stride + 1)
    out = np.zeros((out_h, out_w))
    
    for i_,i in enumerate(range(0, height-kernel_size_h+1, stride)):
        for j_,j in enumerate(range(0, width-kernel_size_w+1, stride)):
            sample = im[i:i+kernel_size_h, j:j+kernel_size_w]
            out[i_, j_] = np.sum(sample*kernel) + bias
    return out

def dilate(x, dilation_h, dilation_w):
    rows, cols = x.shape
    dilated = np.zeros((rows*(1 + dilation_h) - dilation_h, cols*(1 + dilation_w) - dilation_w))
    for i in range(dilated.shape[0]):
        for j in range(dilated.shape[1]):
            dilated[i, j] = x[int(i/(dilation_h+1)), int(j / (dilation_w+1))] \
                            if (i % (dilation_h+1) == 0 and j % (dilation_w+1) == 0) \
                            else 0
    return dilated

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    pad = conv_param['pad']
    stride = conv_param['stride']

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of a convolutional neural network.
    #   Store the output as 'out'.
    #   Hint: to pad the array, you can use the function np.pad.
    # ================================================================ #
    
    a = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)))
    num_kernels = w.shape[0]
    kernel_size_h = w.shape[2]
    kernel_size_w = w.shape[3]
    batch_size, channels, height, width = a.shape
    
    out_h = int((height - kernel_size_h) / stride + 1)
    out_w = int((width - kernel_size_w) / stride + 1)
    
    out = np.zeros((batch_size, num_kernels, out_h, out_w))
    
    for f,kernel in enumerate(w):
        for n,im in enumerate(a):
            bias = b[f]
            conv = np.zeros((out_h, out_w))
            for c in range(0, channels):
                conv += conv2d(im[c], kernel[c], stride, 0)
            out[n, f, :, :] = conv+bias
                
#     print("Input sizes:")
#     print(f"{x.shape=}")
#     print(f"{w.shape=}")
#     print(f"{b.shape=}")
#     print(f"{out.shape=}")
    
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None

    N, F, out_height, out_width = dout.shape
    x, w, b, conv_param = cache

    stride, pad = [conv_param['stride'], conv_param['pad']]
    xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
    num_filts, _, f_height, f_width = w.shape

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of a convolutional neural network.
    #   Calculate the gradients: dx, dw, and db.
    # ================================================================ #
    num_kernels = w.shape[1]
    kernel_size_h = w.shape[2]
    kernel_size_w = w.shape[3]
    batch_size, channels, height, width = x.shape
    
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    for n,im in enumerate(x):
        for f,kernel in enumerate(w):
            for c in range(0, channels):
                dx_ = conv2d(
                    np.pad(
                        dilate(dout[n, f], stride-1, stride-1),
                        ((kernel_size_h-1,kernel_size_h-1),
                        (kernel_size_w-1,kernel_size_w-1))
                    ),
                    np.flip(kernel[c])
                )
                dx[n, c, :, :] += dx_[pad:pad+height, pad:pad+width]
                
                dw_ = conv2d(
                    np.pad(x[n, c], ((pad,pad),(pad,pad))),
                    dilate(dout[n, f], stride-1, stride-1)
                )
                dw[f, c, :, :] += dw_
    
    
    db = np.sum(dout, axis=(0,2,3))
    
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the max pooling forward pass.
    # ================================================================ #
    
    N, C, H, W = x.shape
    R = pool_param['pool_height']
    S = pool_param['pool_width']
    stride = pool_param['stride']
    
    out = np.zeros((N, C, int((H-R)/stride + 1), int((W-S)/stride + 1)))
    for n,im in enumerate(x):
        for c,channel in enumerate(im):
            for i_,i in enumerate(range(0, H-R+1, stride)):
                for j_,j in enumerate(range(0, W-S+1, stride)):
                    out[n, c, i_, j_] = np.max(channel[i:i+R, j:j+S])

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ # 
    cache = (x, pool_param)
    return out, cache

def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    x, pool_param = cache
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the max pooling backward pass.
    # ================================================================ #

    N, C, H, W = x.shape
    R = pool_param['pool_height']
    S = pool_param['pool_width']
    stride = pool_param['stride']
    
    out = np.zeros((N, C, int((H-R)/stride + 1), int((W-S)/stride + 1)))
    dx = np.zeros_like(x)
    
    for n,im in enumerate(x):
        for c,channel in enumerate(im):
            for i_,i in enumerate(range(0, H-R+1, stride)):
                for j_,j in enumerate(range(0, W-S+1, stride)):
                    mask = np.ones_like(channel)*(-np.inf)
                    mask[i:i+R, j:j+S] = channel[i:i+R, j:j+S]
                    dx[n, c, *np.unravel_index(np.argmax(mask), mask.shape)] += dout[n, c, i_, j_]

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ # 

    return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the spatial batchnorm forward pass.
    #
    #   You may find it useful to use the batchnorm forward pass you 
    #   implemented in HW #4.
    # ================================================================ #

    pass

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ # 

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the spatial batchnorm backward pass.
    #
    #   You may find it useful to use the batchnorm forward pass you 
    #   implemented in HW #4.
    # ================================================================ #

    pass

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ # 

    return dx, dgamma, dbeta