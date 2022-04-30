''' Solve deblur problem by Proximal '''
import sys

from proximal.utils.utils import *
from proximal.halide.halide import *
from proximal.lin_ops import *
from proximal.prox_fns import *
from proximal.algorithms import *

import numpy as np
import time


########################################################
def TVL1(img_in, k_in, max_iter, lamb_da):
    """ TVL1 deblur
            Args:
                img_in (uint8 ndarray, shape(height, width, ch)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): Blur kernel
                max_iter (int): Total iteration count
                lamb_da (float): TVL1 variable
                
            Returns:
                TVL1_result (uint8 ndarray, shape(height, width, ch)): Deblurred image
                
            Todo:
                TVL1 deblur
    """
    

    img = img_in/255.0
    K = k_in/255.0
    K = K/K.sum()
    
    
    K_rgb = np.zeros((K.shape[0], K.shape[1], 3))
    K_rgb[:,:,0] = K
    K_rgb[:,:,1] = K
    K_rgb[:,:,2] = K
    K = K_rgb
    
    # test the solver with some sparse gradient deconvolution
    eps_abs_rel = 1e-3
    test_solver = 'pc'

    
    #%% rgb channels
    TVL1_result = Variable(img.shape)
    
    # model the problem by proximal
    prob = Problem(norm1(conv(K,TVL1_result, dims=2) - img) + lamb_da * group_norm1( grad(TVL1_result, dims = 2), [3] ) + nonneg(TVL1_result)) # formulate problem
    
    # solve the problem
    result = prob.solve(verbose=True,solver=test_solver,x0=img,eps_abs=eps_abs_rel, eps_rel=eps_abs_rel,max_iters=max_iter) # solve problem
    TVL1_result = TVL1_result.value
    
    # output color image
    TVL1_result = np.clip(TVL1_result*255+0.5,0,255).astype('uint8')
    return TVL1_result



########################################################
def TVL2(img_in, k_in, max_iter, lamb_da):
    """ TVL2 deblur
            Args:
                img_in (uint8 ndarray, shape(height, width, ch)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): blur kernel
                max_iter (int): total iteration count
                lamb_da (float): TVL2 variable
                
            Returns:
                TVL2_result (uint8 ndarray, shape(height, width, ch)): deblurred image
                
            Todo:
                TVL2 deblur
    """
    img = img_in/255.0
    K = k_in/255.0
    K = K/K.sum()
    
    
    K_rgb = np.zeros((K.shape[0], K.shape[1], 3))
    K_rgb[:,:,0] = K
    K_rgb[:,:,1] = K
    K_rgb[:,:,2] = K
    K = K_rgb
    
    # test the solver with some sparse gradient deconvolution
    eps_abs_rel = 1e-3
    test_solver = 'pc'

    
    #%% rgb channels
    TVL2_result = Variable(img.shape)
    
    # model the problem by proximal
    prob = Problem(sum_squares(conv(K,TVL2_result, dims=2) - img) + lamb_da * group_norm1( grad(TVL2_result, dims = 2), [3] ) + nonneg(TVL2_result)) # formulate problem
    
    # solve the problem
    result = prob.solve(verbose=True,solver=test_solver,x0=img,eps_abs=eps_abs_rel, eps_rel=eps_abs_rel,max_iters=max_iter) # solve problem
    TVL2_result = TVL2_result.value
    
    # output color image
    TVL2_result = np.clip(TVL2_result*255+0.5,0,255).astype('uint8')
    return TVL2_result




########################################################
def TVpoisson(img_in, k_in, max_iter, lamb_da):
    """ TVLpoisson deblur
            Args:
                img_in (uint8 ndarray, shape(height, width, ch)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): blur kernel
                max_iter (int): total iteration count
                lamb_da (float): TVpoisson variable
                
            Returns:
                TVpoisson_result (uint8 ndarray, shape(height, width, ch)): deblurred image
                
            Todo:
                TVpoisson deblur
    """
    img = img_in/255.0
    K = k_in/255.0
    K = K/K.sum()
    
    
    K_rgb = np.zeros((K.shape[0], K.shape[1], 3))
    K_rgb[:,:,0] = K
    K_rgb[:,:,1] = K
    K_rgb[:,:,2] = K
    K = K_rgb
    
    # test the solver with some sparse gradient deconvolution
    eps_abs_rel = 1e-3
    test_solver = 'pc'

    
    #%% rgb channels
    TVpoisson_result = Variable(img.shape)
    
    # model the problem by proximal
    prob = Problem(poisson_norm(conv(K,TVpoisson_result, dims=2), img) + lamb_da * group_norm1( grad(TVpoisson_result, dims = 2), [3] ) + nonneg(TVpoisson_result)) # formulate problem
    
    # solve the problem
    result = prob.solve(verbose=True,solver=test_solver,x0=img,eps_abs=eps_abs_rel, eps_rel=eps_abs_rel,max_iters=max_iter) # solve problem
    TVpoisson_result = TVpoisson_result.value
    
    # output color image
    TVpoisson_result = np.clip(TVpoisson_result*255+0.5,0,255).astype('uint8')
    return TVpoisson_result