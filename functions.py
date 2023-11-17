import numpy as np
import scipy.spatial.distance as distfuncs
import scipy.special as special
import numba as nb

def kernel_helmholtz_3d(p1, p2, wave_num):
    # from Bessel function of 0th order solution
    distance= np.sqrt(np.sum((np.expand_dims(p1,1) - np.expand_dims(p2,0))**2, axis=-1))
    return np.sinc(np.expand_dims(distance,0) * wave_num.reshape(-1, 1,1) / np.pi)

def kernel_dir_3d(p1, p2, wave_num, dir_vec, beta):
    angle = 1j * beta * dir_vec.reshape((1,-1,1,1,dir_vec.shape[-1]))
    pos = wave_num.reshape((-1,1,1,1,1)) * (p1.reshape((1,1,-1,1,p1.shape[-1])) - p2.reshape((1,1,1,-1,p2.shape[-1])))

    return np.sinc(np.sqrt(np.sum((angle - pos)**2, axis=-1)) / np.pi)

def get_kernel_params(func, reg, output, data, *wave_num):
    K = func(data, data, *wave_num)
    soln = K + reg*np.eye(K.shape[-1])

    #print(np.shape(K))
    #kappa = np.transpose(func(output, data, *wave_num))
    kappa = np.moveaxis(func(output, data, *wave_num), -1, -2)
    #print(np.shape(soln))
    #print(np.shape(kappa))
    #print(np.shape(np.transpose(soln)))
    #params = np.transpose(np.linalg.solve(np.transpose(soln), kappa))
    params = np.moveaxis(np.linalg.solve(np.moveaxis(soln, -1, -2), kappa), -1, -2)

    return params


def diffuse_est(p, pos, pos_til, wave_num, reg):
    filter = get_kernel_params(kernel_helmholtz_3d, reg, pos_til, pos, wave_num)
    #p = np.transpose(p)
    #print(np.shape(filter))
    #print(np.shape(p))
    p_est  = filter @ p[:,:,None]
    
    return np.squeeze(p_est, axis=-1)

def directional_est(p, pos, pos_til, wave_num, reg, direction, direction_param):
    filter = get_kernel_params(kernel_dir_3d, reg, pos_til, pos, wave_num, direction, direction_param)[:,0,:,:]
    #p = np.transpose(p)
    p_est = filter @ p[:,:,None]

    return np.squeeze(p_est, axis=-1)


def freq_bins(num_freqs, fs):
    return (fs/num_freqs) * np.arange(num_freqs // 2 + 1)


def generate_pressure_data(fs):
    rng = np.random.default_rng(10)
    length = 1
    step = 0.1
    num_mic = 10
    num_freqs = 128

    mic_pos = np.zeros((num_mic, 3))
    mic_pos[:,:2] = rng.uniform(low=-length/2, high=length/2, size=(num_mic,2))
    #src_pos = np.array([[3,0.05,-0.05]])

    v_coor   = np.arange(-length/2, length/2, step)
    h_coor   = np.arange(-length/2, length/2, step)
    eval_pos = np.array(np.meshgrid(v_coor,h_coor)).T.reshape(-1,2)
    eval_pos = np.c_[eval_pos, np.zeros(np.shape(eval_pos)[0])]
    num_eval_pos = eval_pos.shape[0]

    freqs = freq_bins(num_freqs, fs)
    num_real_freqs = freqs.shape[0]

    p_eval = rng.uniform(low=0, high=1, size=(num_eval_pos, num_real_freqs))
    p_mic  = rng.uniform(low=0, high=1, size=(num_mic, num_real_freqs))

    return mic_pos, np.transpose(p_mic), eval_pos, np.transpose(p_eval), freqs




    