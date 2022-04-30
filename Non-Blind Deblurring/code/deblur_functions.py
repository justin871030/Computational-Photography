''' Functions in deblur flow '''

import numpy as np
import cv2 as cv
from scipy import ndimage, signal
from scipy.signal import convolve2d

import sys
DBL_MIN = sys.float_info.min
from numpy.fft import rfft2, irfft2
########################################################
def Wiener_deconv(img_in, k_in, SNR_F):
    """ Wiener deconvolution
            Args:
                img_in (uint8 ndarray, shape(height, width, ch)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): Blur kernel
                SNR_F (float): Wiener deconvolution parameter
            Returns:
                Wiener_result (uint8 ndarray, shape(height, width, ch)): Wiener-deconv image
                
            Todo:
                Wiener deconvolution
    """

    k_in = k_in.astype('float')
    k_normal = k_in / k_in.sum()

    k_normal_pad = np.zeros(img_in.shape[:2])
    k_normal_pad[tuple([slice(0, s) for s in k_normal.shape])] = k_normal
    for axis, axis_size in enumerate(k_in.shape):
        k_normal_pad = np.roll(k_normal_pad, shift=-int(np.floor(axis_size / 2)), axis=axis) # preprocessing of 2D-DFT on blur kernel

    kernel = rfft2(k_normal_pad) # K(f) = FFT{K}

    WD_result = np.zeros_like(img_in, dtype='float')
    img_in = img_in.astype('float')
    img_in /= 255.0

    for channel in range(img_in.shape[-1]):
        wiener_filter = np.conj(kernel) / (np.abs(kernel) ** 2 + 1/SNR_F)             # wiener_filter = I(f)/B(f) = K(f)* / (|K(f)|^2 + 1/SNR_F)
        WD_result[:,:,channel] = irfft2(wiener_filter * rfft2(img_in[:,:,channel]))   # IFFT{I(f)} = IFFT{wiener_filter * B(f)},  B(f) = FFT(B)

    WD_result = np.round(np.clip(WD_result, 0, 1) * 255).astype('uint8')

    return WD_result

########################################################
def RL(img_in, k_in, max_iter):
    """ RL deconvolution
            Args:
                img_in (uint8 ndarray, shape(height, width, ch)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): blur kernel
                max_iter (int): total iteration count
                
            Returns:
                RL_result (uint8 ndarray, shape(height, width, ch)): RL-deblurred image
                
            Todo:
                RL deconvolution
    """
    
    k_in = k_in.astype('float')
    k_normal = k_in / k_in.sum()
    
    k_star = np.flip(k_normal)  # K*(i, j) = K(−i, −j)
    
    RL_result = np.zeros_like(img_in, dtype='float')
    img_in = img_in.astype('float')
    img_in /= 255

    for channel in range(img_in.shape[-1]):
        I0 = img_in[:,:,channel]
        I_t = I0
        
        for _ in range(max_iter):
            B_d = I0 / signal.convolve2d(I_t, k_normal, boundary='symm', mode='same')
            I_t = I_t * signal.convolve2d(B_d, k_star, boundary='symm', mode='same')
        
        RL_result[:,:,channel] = I_t
    
    RL_result = np.round(np.clip(RL_result, 0, 1) * 255).astype('uint8')
    
    return RL_result



########################################################
def BRL(img_in, k_in, max_iter, lamb_da, sigma_r, rk):
    """ BRL deconvolution
            Args:
                img_in (uint8 ndarray, shape(height, width, ch)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): Blur kernel
                max_iter (int): Total iteration count
                lamb_da (float): BRL parameter
                sigma_r (float): BRL parameter
                rk (int): BRL parameter
                
            Returns:
                BRL_result (uint8 ndarray, shape(height, width, ch)): BRL-deblurred image
                
            Todo:
                BRL deconvolution
    """

    r_omega = 0.5 * rk           
    sigma_s = (r_omega/3)**(2)  
    
    k_in = k_in.astype('float')
    k_in /= k_in.sum()

    k_star = np.flip(k_in)
    
    BRL_result = np.zeros_like(img_in, dtype='float')
    img_in = img_in.astype('float')
    img_in /= 255

    for channel in range(img_in.shape[-1]):
        I0 = img_in[:,:,channel]
        I_t = I0

        for _ in range(max_iter):
            B_d = I0 / signal.convolve2d(I_t, k_in, boundary='symm', mode='same')
            grad_EB = dEB(I_t, sigma_s, sigma_r, r_omega)
            I_t = I_t/(1 + lamb_da*grad_EB) * signal.convolve2d(B_d, k_star, boundary='symm', mode='same') # I_t1 = I_t/scale x (K* conv (B/divider))
        
        BRL_result[:,:,channel] = I_t 
    
    BRL_result = np.round(np.clip(BRL_result, 0, 1) * 255).astype('uint8')

    return BRL_result
    
########################################################
def dEB(I, sigma_s, sigma_r, r_omega):
    """
        Gradient EB
            Args:
                I (uint8 ndarray, shape(height, width)): A channel of a blurred image
                sigma_s (float): BRL parameter
                sigma_r (float): BRL parameter
                r_omega (int): half of kernel size
            Returns:
                dEB (float np.array): gradient of EB
            Todo:
                gradient of EB
            
    """
    kernel_w = int(r_omega*2 + 1)
    half_w = int(r_omega)
    x, y = np.mgrid[-half_w:half_w+1, -half_w:half_w+1]
    gauFir = np.exp(-(x**2 + y**2)/(2*(sigma_s)))
    
    I_pad = np.pad(I, ((half_w, half_w), (half_w, half_w)), 'symmetric')
    
    h = I.shape[0]                                        
    w = I.shape[1]
    dEB = np.zeros_like(I)
    
    for i in range(h):
        for j in range(w):
            diff = I_pad[i+half_w, j+half_w] - I_pad[i:i+kernel_w, j:j+kernel_w]
            dEB[i, j] = 2 * np.sum(gauFir * np.exp(-(diff**2)/(2*(sigma_r))) * (diff/sigma_r))
    
    return dEB

########################################################
def RL_energy(img_in, k_in, I_in):
    """ RL Energy
            Args:
                img_in (uint8 ndarray, shape(height, width, ch)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): Blur kernel
                I_in (uint8 ndarray, shape(height, width, ch)): Deblurred image
                
            Returns:
                RL_energy (float): RL_energy
                
            Todo:
                Calculate RL energy
    """
    
    k_in = k_in.astype('float')
    k_in /= k_in.sum()
    
    img_in = img_in.astype('float')
    I_in = I_in.astype('float')
    
    img_in /= 255
    I_in /= 255
    
    energy = 0
    for channel in range(img_in.shape[-1]):
        I_conv_K = signal.convolve2d(I_in[:,:,channel], k_in, boundary='symm', mode='same')
        energy += np.sum(I_conv_K - img_in[:,:,channel] * np.log(I_conv_K))
    
    return energy

########################################################
def BRL_energy(img_in, k_in, I_in, lamb_da, sigma_r, rk):
    """ BRL Energy
            Args:
                img_in (uint8 ndarray, shape(height, width, ch)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): Blur kernel
                I_in (uint8 ndarray, shape(height, width, ch)): Deblurred image
                lamb_da (float): BRL parameter
                sigma_r (float): BRL parameter
                rk (int): BRL parameter
                
            Returns:
                BRL_energy (float): BRL_energy
                
            Todo:
                Calculate BRL energy
    """

    r_omega = 0.5 * rk      
    sigma_s = (r_omega/3)**(2)  
    
    k_in = k_in.astype('float')
    k_in /= k_in.sum()
    
    img_in = img_in.astype('float')
    I_in = I_in.astype('float')
    
    img_in /= 255.0
    I_in /= 255.0
    
    BRL_energy = 0
    for channel in range(img_in.shape[-1]):
        I = I_in[:,:,channel]
        I_cov_K = signal.convolve2d(I, k_in, boundary='symm', mode='same')
        BRL_energy += np.sum(I_cov_K - img_in[:,:,channel] * np.log(I_cov_K)) + lamb_da*EB(I, sigma_s, sigma_r, r_omega)

    return BRL_energy

########################################################
def EB(I, sigma_s, sigma_r, r_omega):
    """
        EB
            Args:
                I (uint8 ndarray, shape(height, width)): A channel of a blurred image
                sigma_s (float): BRL parameter
                sigma_r (float): BRL parameter
                r_omega (int): half of kernel size --> kernel_w // 2
            Returns:
                EB (float np.array): EB (energy)
            Todo:
                BRL EB for a channel
    """
    kernel_w = int(r_omega*2 + 1)
    half_w = int(r_omega)
    x, y = np.mgrid[-half_w:half_w+1, -half_w:half_w+1]
    gauFir = np.exp(-(x**2 + y**2)/(2*(sigma_s)))
    
    I_pad = np.pad(I, ((half_w, half_w), (half_w, half_w)), 'symmetric')
    
    h = I.shape[0]                                        
    w = I.shape[1]
    e = np.zeros_like(I)
    
    for i in range(h):
        for j in range(w):
            e[i, j] = np.sum(gauFir * (1 - np.exp(-((I_pad[i+half_w, j+half_w] - I_pad[i:i+kernel_w, j:j+kernel_w])**2)/(2*(sigma_r)))))

    return np.sum(e)
