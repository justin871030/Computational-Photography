''' Functions in HDR flow '''

import os
import cv2 as cv
import numpy as np

Z = 256  # intensity levels
Z_max = 255
Z_min = 0
Z_avg = (Z_max + Z_min) / 2
w = np.array([z - Z_min if z <= Z_avg else Z_max - z for z in range(Z)])
gamma = 2.2

import matplotlib.pyplot as plt


def ReadImg(path, flag=1):
    img = cv.imread(path, flag)  # flag = 1 means to load a color image
    img = np.transpose(img, (2,0,1))
    return img


def SaveImg(img, path):
    img = np.transpose(img, (1,2,0))
    cv.imwrite(path, img)
    
    
def LoadExposures(source_dir):
    """ load bracketing images folder

    Args:
        source_dir (string): folder path containing bracketing images and a image_list.txt file
                             image_list.txt contains lines of image_file_name, exposure time, ... 
    Returns:
        img_list (uint8 ndarray, shape (N, ch, height, width)): N bracketing images (3 channel)
        exposure_times (list of float, size N): N exposure times
    """
    filenames = []
    exposure_times = []
    f = open(os.path.join(source_dir, 'image_list.txt'))
    for line in f:
        if (line[0] == '#'):
            continue
        (filename, exposure, *_) = line.split()
        filenames += [filename]
        exposure_times += [float(exposure)]
    img_list = [ReadImg(os.path.join(source_dir, f)) for f in filenames]
    img_list = np.array(img_list)
    
    return img_list, exposure_times


def PixelSample(img_list):
    """ Sampling

    Args:
        img_list (uint8 ndarray, shape (N, ch, height, width))
        
    Returns:
        sample (uint8 ndarray, shape (N, ch, height_sample_size, width_sample_size))
    """
    # trivial periodic sample
    sample = img_list[:, :, ::64, ::64]
    
    return sample


def EstimateResponse(img_samples, etime_list, lambda_=50):
    """ Estimate camera response for bracketing images

    Args:
        img_samples (uint8 ndarray, shape (N, height_sample_size, width_sample_size)): N bracketing sampled images (1 channel)
        etime_list (list of float, size N): N exposure times
        lambda_ (float): Lagrange multiplier (Defaults to 50)
    
    Returns:
        response (float ndarray, shape (256)): response map
    """

    ''' TODO '''
    Z = img_samples.reshape(img_samples.shape[0], -1)
    n = 256
    I = Z.shape[1] # number of sampled pixels
    J = Z.shape[0] # number of of images
    M = I*J + (n-2) + 1
    N = n + I
    A = np.zeros((M, N), dtype=np.float32)
    b = np.zeros((M, 1), dtype=np.float32)
    ln_t = np.log(etime_list)

    k = 0
    for i in range(I):
        for j in range(J):
            z = Z[j][i]
            wij = w[z]
            A[k][z] = wij
            A[k][n+i] = -wij
            b[k] = wij*ln_t[j]
            k += 1

    A[k][128] = 1
    k += 1

    for i in range(1, n-1):
        A[k][i-1]   = lambda_ * w[i]
        A[k][i] = -2 * lambda_ * w[i]
        A[k][i+1] = lambda_ * w[i]
        k += 1
    
    x = np.linalg.lstsq(A, b)[0]
    response = x[:256].reshape(-1)
    
    return response



def ConstructRadiance(img_list, response, etime_list):
    """ Construct radiance map from brackting images

    Args:
        img_list (uint8 ndarray, shape (N, height, width)): N bracketing images (1 channel)
        response (float ndarray, shape (256)): response map
        etime_list (list of float, size N): N exposure times
    
    Returns:
        radiance (float ndarray, shape (height, width)): radiance map
    """

    ''' TODO '''
    Z = img_list.reshape(img_list.shape[0], -1)
    sum_E = ((response[Z].reshape(Z.shape) - np.log(etime_list).reshape(-1, 1))*w[Z]).sum(axis=0)
    sum_w = w[Z].sum(axis=0)
    sum_w[sum_w <= 0] = 1
    E = np.exp(sum_E/sum_w)

    return E.reshape(img_list[0].shape)


def CameraResponseCalibration(src_path, lambda_):
    img_list, exposure_times = LoadExposures(src_path)
    radiance = np.zeros_like(img_list[0], dtype=np.float32)
    pixel_samples = PixelSample(img_list)

    for ch in range(3):
        response = EstimateResponse(pixel_samples[:,ch,:,:], exposure_times, lambda_)
        radiance[ch,:,:] = ConstructRadiance(img_list[:,ch,:,:], response, exposure_times)
    #     print(lE.shape)
    #     plt.figure()
    #     plt.imshow(lE.reshape(12,8), cmap='gray')
    #     plt.figure()
    #     plt.imshow(radiance[ch], cmap='gray')
    # plt.show()
        
    return radiance


def WhiteBalance(src, y_range, x_range):
    """ White balance based on Known to be White(KTBW) region

    Args:
        src (float ndarray, shape (ch, height, width)): source radiance
        y_range (tuple of 2 int): location range in y-dimension
        x_range (tuple of 2 int): location range in x-dimension
        
    Returns:
        result (float ndarray, shape (ch, height, width))
    """
   
    ''' TODO '''
    
    B_G_R_avg = src[:, y_range[0]:y_range[1], x_range[0]:x_range[1]].reshape(3, -1).mean(1)
    B_G_R_avg /= B_G_R_avg[2]
    
    return src / B_G_R_avg.reshape(-1, 1, 1)


def GlobalTM(src, scale=1.0):
    """ Global tone mapping

    Args:
        src (float ndarray, shape (ch, height, width)): source radiance image
        scale (float): scaling factor (Defaults to 1.0)
    
    Returns:
        result(uint8 ndarray, shape (ch, height, width)): result HDR image
    """
    
    ''' TODO '''
    # search for the maximum value in R, G, B channels separately
    X_max = np.ones_like(src) * src.max(axis=(1,2)).reshape(-1, 1, 1)
    
    ldr = (2**(scale*np.log2(src/X_max)+np.log2(X_max)))**(1/gamma)
    # ldr /= ldr.mean()
    # ldr -= 1
    # ldr = 1/(1+np.exp(-ldr*0.5))
    # import pdb; pdb.set_trace()
    ldr = ldr.clip(0, 1)
    
    ldr = np.round(ldr * 255)
    
    return ldr.astype("uint8")



def LocalTM(src, imgFilter, scale=3.0):
    """ Local tone mapping

    Args:
        src (float ndarray, shape (ch, height, width)): source radiance image
        imgFilter (function): filter function with preset parameters
        scale (float): scaling factor (Defaults to 3.0)
    
    Returns:
        result(uint8 ndarray, shape (ch, height, width)): result HDR image
    """
    
    ''' TODO '''
    I = np.average(src, axis=0)              # I = avg(R, G, B)
    Cx = src/I                               # Cx = X/I, for X ∈ {R,G,B}
    
    L = np.log2(I)                           # L = log2(I)
    
    LB = imgFilter(L)
    LD = L - LB

    Lmax = LB.max()                          # Lmax = max LB(i,j)
    Lmin = LB.min()                          # Lmin = min LB(i,j)
    
    LB_ = (LB - Lmax)*(scale)/(Lmax - Lmin)  # L′B = (LB − Lmax) * scale/(Lmax − Lmin)
    
    I_ = 2**(LB_ + LD)                       # I′ = 2^(L′B+LD) 
    
    C = Cx * I_                              # Cx*I′, for X ∈ {R,G,B}
    
    C = C**(1/gamma)

    C = C.clip(0,1)
    
    C = np.round(C * 255)
    
    return C.astype("uint8")

def GaussianFilter(src, N=35, sigma_s=100):
    """ Gaussian filter

    Args:
        src (float ndarray, shape (height, width)): source intensity
        N (int): window size of the filter (Defaults to 35)
                 filter indices span [-N/2, N/2]
        sigma_s (float): standard deviation of Gaussian filter (Defaults to 100)
    
    Returns:
        result (float ndarray, shape (height, width))
    """
    
    ''' TODO '''
    assert N % 2
    n = N // 2
    ax = np.linspace(-N, N, N)
    gauss_1d = np.exp(-0.5 * np.square(ax) / np.square(sigma_s))
    gauss_2d = np.outer(gauss_1d, gauss_1d)
    
    L_pad = np.pad(src, ((n, n), (n, n)), 'symmetric')
    
    LB = np.zeros_like(src)

    gauss_2d_sum = np.sum(gauss_2d)
    
    for i in range(LB.shape[0]):
        for j in range(LB.shape[1]):
            ele_wise = L_pad[i:i+N, j:j+N] * gauss_2d
            LB[i, j] = np.sum(ele_wise) / gauss_2d_sum
            
    return LB


def BilateralFilter(src, N=35, sigma_s=100, sigma_r=0.8):
    """ Bilateral filter

    Args:
        src (float ndarray, shape (height, width)): source intensity
        N (int): window size of the filter (Defaults to 35)
                 filter indices span [-N/2, N/2]
        sigma_s (float): spatial standard deviation of bilateral filter (Defaults to 100)
        sigma_r (float): range standard deviation of bilateral filter (Defaults to 0.8)
    
    Returns:
        result (float ndarray, shape (height, width))
    """
    
    ''' TODO '''
    
    assert N % 2
    n = N // 2
    ax = np.linspace(-N, N, N)
    gauss_1d = np.exp(-0.5 * np.square(ax) / np.square(sigma_s))
    gauss_2d = np.outer(gauss_1d, gauss_1d)
    
    L_pad = np.pad(src, ((n, n), (n, n)), 'symmetric')
    
    LB = np.zeros_like(src)
    
    for i in range(LB.shape[0]):
        for j in range(LB.shape[1]):
            bilaterFir = gauss_2d * np.exp(-0.5 * (L_pad[i+n, j+n] - L_pad[i:i+N, j:j+N])**2 / (sigma_r**2))
            ele_wise = L_pad[i:i+N, j:j+N] * bilaterFir 
            LB[i, j] = np.sum(ele_wise) / bilaterFir.sum()
            
    return LB
