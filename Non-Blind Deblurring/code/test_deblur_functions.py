''' Test functions in deblur flow'''

import time
import numpy as np
import imageio

from PIL import Image
from deblur_functions import Wiener_deconv, RL, BRL, RL_energy, BRL_energy
from TV_functions import TVL1, TVL2, TVpoisson

INF = float("inf")
import sys
DBL_MIN = sys.float_info.min

### TEST_PAT_SIZE can be assigned to either 'small' or 'large' ###
#-- 'small' stands for small test pattern size
#-- 'large' stands for large test pattern size
# During implementation, it is recommended to set TEST_PAT_SIZE 'small' for quick debugging.
# However, you have to pass the unit test with TEST_PAT_SIZE 'large' to get the full score in each part.
TEST_PAT_SIZE = 'small'



########################################################################
def PSNR_UCHAR3(input_1, input_2, peak=255):
    [row,col,channel] = input_1.shape
    
    if input_1.shape != input_2.shape:
        print ("Warning!! Two image have different shape!!")
        return 0
    
    input_1 = input_1.astype('float')
    input_2 = input_2.astype('float')
    
    mse = ((input_1 - input_2)**2).sum() / float(row * col * channel)
    
    # print('mse: ', mse)
    if mse == 0.0:
        psnr = INF  # avoid divide zero case
    else:
        psnr = 10 * np.log10((255.0 ** 2)/mse)
    
    return psnr

########################################################################
def Evaluate_PSNR(psnr, duration, target_psnr=60.0):
    print(f'    -> processing time = {duration:.2f} sec, PSNR = {psnr} dB')
    
    if(psnr<target_psnr): 
        print('    -> status: \033[1;31;40m fail \033[0;0m ... QQ\n')
    else:
        print('    -> status: \033[1;32;40m pass \033[0;0m !!\n')   
    
    
    
########################################################################
def Evaluate_error(error, duration):
    print(f'    -> processing time = {duration:.2f} sec, error = {error:.4f} %')
    
    if(error>0.05): 
        print('    -> status: \033[1;31;40m fail \033[0;0m ... QQ\n')
    else:
        print('    -> status: \033[1;32;40m pass \033[0;0m !!\n') 
    
    
    
    
########################################################################
def test_Wiener_deconv():
    print ("//--------------------------------------------------------")
    print (f"start Wiener deconvolution, TEST_PAT_SIZE: {TEST_PAT_SIZE}\n")

    # I/O
    img_in = np.asarray(Image.open(f'../data/blurred_image_{TEST_PAT_SIZE}/curiosity_medium.png'))
    k_in = np.asarray(Image.open('../data/kernel/kernel_medium.png'))  
    img_golden = np.asarray(Image.open(f'../golden/golden_{TEST_PAT_SIZE}/curiosity_medium/Wiener_m_SNRF100.0.png'))
    
    # setting 
    SNR_F = 100.0

    # work
    t_start = time.time()
    Wiener_result = Wiener_deconv(img_in, k_in, SNR_F)
    t_end = time.time()

    # store image
    imageio.imwrite(f'../result/result_{TEST_PAT_SIZE}/curiosity_medium/Wiener_m_SNRF100.0.png' , Wiener_result)

    # evaluate
    psnr = PSNR_UCHAR3(Wiener_result, img_golden)
    duration = t_end - t_start
    Evaluate_PSNR(psnr, duration)  





########################################################################
def test_RL_a():
    print ("//--------------------------------------------------------")
    print (f"start RL-(a) deconvolution, TEST_PAT_SIZE: {TEST_PAT_SIZE}\n")
    
    # I/O
    img_in = np.asarray(Image.open(f'../data/blurred_image_{TEST_PAT_SIZE}/curiosity_small.png'))
    k_in = np.asarray(Image.open('../data/kernel/kernel_small.png'))  
    img_golden = np.asarray(Image.open(f'../golden/golden_{TEST_PAT_SIZE}/curiosity_small/RL_s_iter30.png'))
    
    # setting 
    max_iter_RL = 30
    
    # work
    t_start = time.time()
    RL_result = RL(img_in, k_in, max_iter_RL)
    t_end = time.time()

    # store image
    imageio.imwrite(f'../result/result_{TEST_PAT_SIZE}/curiosity_small/RL_s_iter30.png' , RL_result)

    # evaluate
    psnr = PSNR_UCHAR3(RL_result, img_golden)
    duration = t_end - t_start
    Evaluate_PSNR(psnr, duration)  

    
    
########################################################################
def test_RL_b():
    print ("//--------------------------------------------------------")
    print (f"start RL-(b) deconvolution, TEST_PAT_SIZE: {TEST_PAT_SIZE}\n")
    
    # I/O
    img_in = np.asarray(Image.open(f'../data/blurred_image_{TEST_PAT_SIZE}/curiosity_medium.png'))
    k_in = np.asarray(Image.open('../data/kernel/kernel_medium.png'))  
    img_golden = np.asarray(Image.open(f'../golden/golden_{TEST_PAT_SIZE}/curiosity_medium/RL_m_iter45.png'))
    
    # setting 
    max_iter_RL = 45
    
    # work   
    t_start = time.time()
    RL_result = RL(img_in, k_in, max_iter_RL)
    t_end = time.time()

    # store image
    imageio.imwrite(f'../result/result_{TEST_PAT_SIZE}/curiosity_medium/RL_m_iter45.png' , RL_result)
    
    # evaluate
    psnr = PSNR_UCHAR3(RL_result, img_golden)
    duration = t_end - t_start
    Evaluate_PSNR(psnr, duration)  



    
    
########################################################################
def test_RL_energy():
    print ("//--------------------------------------------------------")
    print (f"start RL-energy function, TEST_PAT_SIZE: {TEST_PAT_SIZE}\n")
    
    # I/O
    blur_in = np.asarray(Image.open(f'../data/blurred_image_{TEST_PAT_SIZE}/curiosity_small.png'))
    k_in = np.asarray(Image.open('../data/kernel/kernel_small.png'))  
    img_in = np.asarray(Image.open(f'../golden/golden_{TEST_PAT_SIZE}/curiosity_small/RL_s_iter30.png'))
    
    energy_dict = np.load(f'../golden/energy_dict.npy',allow_pickle='TRUE').item()
    golden = energy_dict[f'{TEST_PAT_SIZE}_RL_a']
    
    
    # work
    t_start = time.time()
    energy = RL_energy(img_in, k_in, blur_in)
    t_end = time.time()

    # evaluate
    print(f'RL energy: {energy}, golden energy: {golden}')
    duration = t_end - t_start
    Evaluate_error( abs((energy-golden)/golden)*100, duration)
    
    
    
    
########################################################################   
def test_BRL_a():
    print ("//--------------------------------------------------------")
    print (f"start BRL-(a) deconvolution, TEST_PAT_SIZE: {TEST_PAT_SIZE}\n")
    
    # I/O
    img_in = np.asarray(Image.open(f'../data/blurred_image_{TEST_PAT_SIZE}/curiosity_small.png'))
    k_in = np.asarray(Image.open('../data/kernel/kernel_small.png'))  
    img_golden = np.asarray(Image.open(f'../golden/golden_{TEST_PAT_SIZE}/curiosity_small/BRL_s_iter15_rk6_si50.00_lam0.030.png'))
    
    # setting
    max_iter_RL = 15
    rk = 6
    sigma_r = 50.0/255/255
    lamb_da = 0.03/255

    # work
    t_start = time.time()
    BRL_result = BRL(img_in, k_in, max_iter_RL, lamb_da, sigma_r, rk)
    t_end = time.time()

    # store image
    imageio.imwrite(f'../result/result_{TEST_PAT_SIZE}/curiosity_small/BRL_s_iter15_rk6_si50.00_lam0.030.png' , BRL_result)

    # evaluate
    psnr = PSNR_UCHAR3(BRL_result, img_golden)
    duration = t_end - t_start
    Evaluate_PSNR(psnr, duration, target_psnr=55.0)   



########################################################################   
def test_BRL_b():
    print ("//--------------------------------------------------------")
    print (f"start BRL-(b) deconvolution, TEST_PAT_SIZE: {TEST_PAT_SIZE}\n")
    
    # I/O
    img_in = np.asarray(Image.open(f'../data/blurred_image_{TEST_PAT_SIZE}/curiosity_small.png'))
    k_in = np.asarray(Image.open('../data/kernel/kernel_small.png'))  
    img_golden = np.asarray(Image.open(f'../golden/golden_{TEST_PAT_SIZE}/curiosity_small/BRL_s_iter15_rk6_si50.00_lam0.060.png'))
    
    # setting
    max_iter_RL = 15
    rk = 6
    sigma_r = 50.0/255/255
    lamb_da = 0.06/255

    # work
    t_start = time.time()
    BRL_result = BRL(img_in, k_in, max_iter_RL, lamb_da, sigma_r, rk)
    t_end = time.time()

    # store image
    imageio.imwrite(f'../result/result_{TEST_PAT_SIZE}/curiosity_small/BRL_s_iter15_rk6_si50.00_lam0.060.png' , BRL_result)
    
    # evaluate
    psnr = PSNR_UCHAR3(BRL_result, img_golden)
    duration = t_end - t_start
    Evaluate_PSNR(psnr, duration, target_psnr=55.0)  




########################################################################   
def test_BRL_c():
    print ("//--------------------------------------------------------")
    print (f"start BRL-(c) deconvolution, TEST_PAT_SIZE: {TEST_PAT_SIZE}\n")
    
    # I/O
    img_in = np.asarray(Image.open(f'../data/blurred_image_{TEST_PAT_SIZE}/curiosity_medium.png'))
    k_in = np.asarray(Image.open('../data/kernel/kernel_medium.png'))  
    img_golden = np.asarray(Image.open(f'../golden/golden_{TEST_PAT_SIZE}/curiosity_medium/BRL_m_iter40_rk12_si25.00_lam0.001.png'))
    
    # setting
    max_iter_RL = 40
    rk = 12
    sigma_r = 25.0/255/255
    lamb_da = 0.001/255

    # work
    t_start = time.time()
    BRL_result = BRL(img_in, k_in, max_iter_RL, lamb_da, sigma_r, rk)
    t_end = time.time()

    # store image
    imageio.imwrite(f'../result/result_{TEST_PAT_SIZE}/curiosity_medium/BRL_m_iter40_rk12_si25.00_lam0.001.png' , BRL_result)
    
    # evaluate
    psnr = PSNR_UCHAR3(BRL_result, img_golden)
    duration = t_end - t_start
    Evaluate_PSNR(psnr, duration, target_psnr=55.0)    
    
    
########################################################################   
def test_BRL_d():
    print ("//--------------------------------------------------------")
    print (f"start BRL-(d) deconvolution, TEST_PAT_SIZE: {TEST_PAT_SIZE}\n")
    
    # I/O
    img_in = np.asarray(Image.open(f'../data/blurred_image_{TEST_PAT_SIZE}/curiosity_medium.png'))
    k_in = np.asarray(Image.open('../data/kernel/kernel_medium.png'))  
    img_golden = np.asarray(Image.open(f'../golden/golden_{TEST_PAT_SIZE}/curiosity_medium/BRL_m_iter40_rk12_si25.00_lam0.006.png'))
    
    # setting
    max_iter_RL = 40
    rk = 12
    sigma_r = 25.0/255/255
    lamb_da = 0.006/255
    
    # work
    t_start = time.time()
    BRL_result = BRL(img_in, k_in, max_iter_RL, lamb_da, sigma_r, rk)
    t_end = time.time()
    psnr = PSNR_UCHAR3(BRL_result, img_golden)

    # store image
    imageio.imwrite(f'../result/result_{TEST_PAT_SIZE}/curiosity_medium/BRL_m_iter40_rk12_si25.00_lam0.006.png' , BRL_result)
    
    # evaluate
    psnr = PSNR_UCHAR3(BRL_result, img_golden)
    duration = t_end - t_start
    Evaluate_PSNR(psnr, duration, target_psnr=55.0)    
    
########################################################################
def test_BRL_energy():
    print ("//--------------------------------------------------------")
    print (f"start BRL-energy function, TEST_PAT_SIZE: {TEST_PAT_SIZE}\n")
    
    # I/O
    blur_in = np.asarray(Image.open(f'../data/blurred_image_{TEST_PAT_SIZE}/curiosity_small.png'))
    k_in = np.asarray(Image.open('../data/kernel/kernel_small.png'))  
    img_in = np.asarray(Image.open(f'../golden/golden_{TEST_PAT_SIZE}/curiosity_small/BRL_s_iter15_rk6_si50.00_lam0.030.png'))
    
    energy_dict = np.load('../golden/energy_dict.npy',allow_pickle='TRUE').item()
    golden = energy_dict[f'{TEST_PAT_SIZE}_BRL_a']
    

    # setting
    rk = 6
    sigma_r = 50.0/255/255
    lamb_da = 0.03/255

    # work
    t_start = time.time()
    energy = BRL_energy(img_in, k_in, blur_in, lamb_da, sigma_r, rk)
    t_end = time.time()
    
    # evaluate
    print(f'BRL energy: {energy}, golden energy: {golden}')
    duration = t_end - t_start
    Evaluate_error( abs((energy-golden)/golden)*100, duration)
    
    
    
    

########################################################################
def test_TVL1():    
    print ("//--------------------------------------------------------")
    print (f"start TVL1, TEST_PAT_SIZE: {TEST_PAT_SIZE}\n")
    
    # I/O
    img_in = np.asarray(Image.open(f'../data/blurred_image_{TEST_PAT_SIZE}/curiosity_medium.png'))
    k_in = np.asarray(Image.open('../data/kernel/kernel_medium.png'))  
    img_golden = np.asarray(Image.open(f'../golden/golden_{TEST_PAT_SIZE}/curiosity_medium/TVL1_m_iter1000_lam0.010.png'))
    
    # setting
    max_iter = 1000
    lamb_da = 0.01
    
    # work
    t_start = time.time()
    TVL1_result = TVL1(img_in, k_in, max_iter, lamb_da)
    t_end = time.time()
    
    # store image
    imageio.imwrite(f'../result/result_{TEST_PAT_SIZE}/curiosity_medium/TVL1_m_iter1000_lam0.010.png' , TVL1_result)
    
    # evaluate
    psnr = PSNR_UCHAR3(TVL1_result, img_golden)
    duration = t_end - t_start
    Evaluate_PSNR(psnr, duration)    
    
    
########################################################################
def test_TVL2():    
    print ("//--------------------------------------------------------")
    print (f"start TVL2, TEST_PAT_SIZE: {TEST_PAT_SIZE}\n")
    
    # I/O
    img_in = np.asarray(Image.open(f'../data/blurred_image_{TEST_PAT_SIZE}/curiosity_medium.png'))
    k_in = np.asarray(Image.open('../data/kernel/kernel_medium.png'))  
    img_golden = np.asarray(Image.open(f'../golden/golden_{TEST_PAT_SIZE}/curiosity_medium/TVL2_m_iter1000_lam0.010.png'))
    
    # setting
    max_iter = 1000
    lamb_da = 0.01
    
    # work
    t_start = time.time()
    TVL2_result = TVL2(img_in, k_in, max_iter, lamb_da)
    t_end = time.time()

    # store image
    imageio.imwrite(f'../result/result_{TEST_PAT_SIZE}/curiosity_medium/TVL2_m_iter1000_lam0.010.png' , TVL2_result)
    
    # evaluate
    psnr = PSNR_UCHAR3(TVL2_result, img_golden)
    duration = t_end - t_start
    Evaluate_PSNR(psnr, duration)    
    

    
########################################################################
def test_TVpoisson():    
    print ("//--------------------------------------------------------")
    print (f"start TVpoisson, TEST_PAT_SIZE: {TEST_PAT_SIZE}\n")
    
    # I/O
    img_in = np.asarray(Image.open(f'../data/blurred_image_{TEST_PAT_SIZE}/curiosity_medium.png'))
    k_in = np.asarray(Image.open('../data/kernel/kernel_medium.png'))  
    img_golden = np.asarray(Image.open(f'../golden/golden_{TEST_PAT_SIZE}/curiosity_medium/TVpoisson_m_iter1000_lam0.010.png'))
    
    # setting
    max_iter = 1000
    lamb_da = 0.01
    
    # work
    t_start = time.time()
    TVpoisson_result = TVpoisson(img_in, k_in, max_iter, lamb_da)
    t_end = time.time()

    # store image
    imageio.imwrite(f'../result/result_{TEST_PAT_SIZE}/curiosity_medium/TVpoisson_m_iter1000_lam0.010.png' , TVpoisson_result)
    
    # evaluate
    psnr = PSNR_UCHAR3(TVpoisson_result, img_golden)
    duration = t_end - t_start
    Evaluate_PSNR(psnr, duration)    
        
    
    
########################################################################
if __name__ == '__main__':

    ## (1) Wiener part
    
    test_Wiener_deconv()
    
    
    # ## (2) RL part
    
    test_RL_a()
    test_RL_b()
    test_RL_energy()
    
    
    # ## (3) BRL part
    
    test_BRL_a()
    test_BRL_b()
    test_BRL_c()
    test_BRL_d()
    test_BRL_energy()


    ## (4) Total variation part
    
    test_TVL1() # already done function for ref !
    test_TVL2()
    test_TVpoisson()
    
