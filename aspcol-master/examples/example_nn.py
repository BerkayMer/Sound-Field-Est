import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.signal as ss
import pathlib
import math

from nnv import NNV
from aspsim.simulator import SimulatorSetup
from aspsim.room      import roomimpulseresponse as rirs
import aspsim.room.region as reg

import aspcol.kernelinterpolation as ki
import aspcol.scatterki as scki
import aspcol.filterdesign as fd
import aspcol.soundfieldestimation as sfe
import aspcol.ir_freefield as irfr
import aspcol.irplot as ip
import aspcol.neuralki as knn

def run_exp():
    samplerate = 1000
    f_idx = 178
    leb_order = 59
    num_digits = 6
    #num_dirs = 400
    #dirs = knn.dir_vecs_fibonacci(num_dirs)
    dirs, integral_weights = knn.dir_vecs_lebedev(leb_order)
    #knn.plot_vecs(dirs) 

    pos_mic, p_mic, pos_eval, p_eval, freqs, fig_folder, sim_info, pos_sct, sct_region, sct_points, side_len, pos_src = generate_data(samplerate, 100)

    #pos_sct = pos_sct.reshape(1,-1)
    wave_num = freqs / sim_info.c
    min_dly = 0
    extra_delay = 20
    frac_dly_len = 2*(min_dly+extra_delay)+1
    pos_sct = pos_sct.reshape(1,-1)

    
    print(f"Scatterer position: {pos_sct[0]}")
    print("Computing sct2mic irs")
    scatter_ir_mic = np.squeeze(rirs.ir_room_image_source_3d(pos_sct, pos_mic, sim_info.room_size, sim_info.room_center, (len(freqs)-1)*2, 0, sim_info.samplerate, sim_info.c))

    print("Computing sct2eval irs")
    scatter_ir_eval = np.squeeze(rirs.ir_room_image_source_3d(pos_sct, pos_eval, sim_info.room_size, sim_info.room_center, (len(freqs)-1)*2, 0, sim_info.samplerate, sim_info.c))


    p_mic_time = np.fft.irfft(np.transpose(p_mic), n=(len(freqs)-1)*2) # 41x16
    #p_sct_mic_time = ss.fftconvolve(p_mic_time[:-1,:], scatter_ir_mic, mode="full", axes=1)
    p_sct_mic_time = ss.fftconvolve(p_mic_time[:,:], scatter_ir_mic, mode="full", axes=1)
    p_sct_mic_time = np.zeros(p_sct_mic_time.shape)

    for idx in range(p_mic_time.shape[0]):
        p_sct_mic_time[idx,:] = ss.fftconvolve(p_mic_time[-1,:], scatter_ir_mic[idx,:], mode="full", axes = 0)
    #p_sct_mic_freq = np.transpose(np.fft.rfft(p_sct_mic_time, n=(len(freqs)-1)*2))
    
    p_sct_time = p_mic_time[-1,:]

    # min_dly offset
    delay_filt = np.zeros(p_mic_time.shape)
    delay_filt[:,extra_delay] = 1
    p_mic_time = ss.fftconvolve(p_mic_time, delay_filt, mode="full", axes=1)

    p_sct_mic_freq = np.transpose(np.fft.rfft(p_sct_mic_time, n=p_sct_mic_time.shape[1]*2))
    p_mic = np.transpose(np.fft.rfft(p_mic_time, n=p_sct_mic_time.shape[1]*2))
    p_mic = p_mic + p_sct_mic_freq
    

    p_eval_time = np.fft.irfft(np.transpose(p_eval), n=(len(freqs)-1)*2) # 400x16
    p_sct_eval_time = ss.fftconvolve(p_eval_time, scatter_ir_eval, mode="full", axes=1) # 400x16

    p_sct_eval_time = np.zeros(p_sct_eval_time.shape)

    for idx in range(p_eval_time.shape[0]):
        p_sct_eval_time[idx,:] = ss.fftconvolve(p_sct_time, scatter_ir_eval[idx,:], mode="full", axes = 0)

    # min_dly offset
    delay_filt = np.zeros(p_eval_time.shape)
    delay_filt[:,extra_delay] = 1
    p_eval_time = ss.fftconvolve(p_eval_time, delay_filt, mode="full", axes=1)

    #p_sct_eval_time = p_sct_eval_time[:,frac_dly_len:]
    p_sct_eval_freq = np.transpose(np.fft.rfft(p_sct_eval_time, n=p_sct_eval_time.shape[1]*2)) # 65x400
    p_eval = np.transpose(np.fft.rfft(p_eval_time, n=p_sct_eval_time.shape[1]*2))
    p_eval_inc = p_eval
    p_eval = p_eval + p_sct_eval_freq

    pnt_idx = 50
    pnt_pos = pos_eval[50]

    ip.IR_time_plot(p_eval_time[pnt_idx,:], scatter_ir_eval[pnt_idx,:], p_mic_time[-1,:], p_sct_eval_time[pnt_idx,:], pos_src, pos_sct[0], pnt_pos, sim_info.samplerate, sim_info.c)
    #breakpoint()

    p_mic = p_mic[:,:-1]
    pos_mic = pos_mic[:-1,:]
    R = side_len/2
    num_freqs = p_sct_eval_time.shape[1]*2
    freqs = fd.get_real_freqs(num_freqs, samplerate)
    wave_num = freqs / sim_info.c
    #breakpoint()

    p_mic[:,sct_region.is_in_region(pos_mic)]   = np.zeros((freqs.shape[0], p_mic[:,sct_region.is_in_region(pos_mic)].shape[1]))
    p_eval[:,sct_region.is_in_region(pos_eval)] = np.zeros((freqs.shape[0], p_eval[:,sct_region.is_in_region(pos_eval)].shape[1]))
    p_eval_inc[:,sct_region.is_in_region(pos_eval)] = np.zeros((freqs.shape[0], p_eval_inc[:,sct_region.is_in_region(pos_eval)].shape[1]))
    p_sct_eval_freq[:,sct_region.is_in_region(pos_eval)] = np.zeros((freqs.shape[0], p_sct_eval_freq[:,sct_region.is_in_region(pos_eval)].shape[1]))

    pos_mic = np.delete(pos_mic, sct_region.is_in_region(pos_mic), axis=0)
    pos_eval = np.delete(pos_eval, sct_region.is_in_region(pos_eval), axis=0)

    p_mic  = p_mic[:, np.any(p_mic, axis=0)]
    p_eval = p_eval[:, np.any(p_eval, axis=0)]
    p_eval_inc = p_eval_inc[:, np.any(p_eval_inc, axis=0)]
    p_sct_eval_freq = p_sct_eval_freq[:, np.any(p_sct_eval_freq, axis=0)]
    

    reg_param = 1e-1
    k = wave_num[f_idx]
    print(f"Frequency: {freqs[f_idx]} Hz")

    # define the network
    # input is the coordinate
    # output should be the weight w(n)
    # then numerical integration of w(n) is kernel
    # use that kernel to estimate pressures
    # the loss function should take those estimations
    model = knn.Kernel_NN()
    '''
    model = nn.Sequential(
        nn.Linear(3,8),
        nn.ReLU(),
        nn.Linear(8,8),
        nn.ReLU(),
        nn.Linear(8,5),
        nn.ReLU(),
        nn.Linear(5,1),
        nn.ReLU()
    )
    '''

    # MODEL TRAINING
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    n_epochs = 50
    batch_size = 20

    
    X = torch.tensor(dirs, dtype=torch.float32)
    p_mic = p_mic[f_idx,:]
    Y = torch.tensor(p_mic, dtype=torch.cfloat)


    loss_func = knn.loss_fn

    prev_mse = 0
    
    
    w_pred = model(X)

    K = knn.get_kernel_lebedev(pos_mic, pos_mic, w_pred, dirs, k, integral_weights)
    K = K.detach().cpu().numpy()
    K_reg = K + reg_param * np.eye(K.shape[-1])

    kappa = knn.get_kernel_lebedev(pos_mic, pos_mic, w_pred, dirs, k, integral_weights)
    kappa = kappa.detach().cpu().numpy()

    est_filt = np.moveaxis(np.linalg.solve(np.moveaxis(K_reg, -1, -2), kappa), -1, -2)
    p_nn = np.squeeze(est_filt @ p_mic[:,None])

    mse = round(10*np.log10(np.mean(np.abs(p_nn - p_mic)**2)), num_digits)
    print(f"MSE without any training: {mse} dB")


    early_stopper = knn.EarlyStopping(limit=2)
    stop_count = 0
    for epochs in range(n_epochs):
        for b_idx in range(0, len(Y), batch_size):
            w_pred = model(X)
            K_pred = knn.get_kernel_lebedev(pos_mic[b_idx:b_idx + batch_size,:], pos_mic[b_idx:b_idx + batch_size,:], w_pred, dirs, k, integral_weights)

            train_loss = loss_func(K_pred, Y[b_idx:b_idx + batch_size], reg_param)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        
        w_pred = model(X)
        K = knn.get_kernel_lebedev(pos_mic, pos_mic, w_pred, dirs, k, integral_weights)
        K = K.detach().cpu().numpy()
        K_reg = K + reg_param * np.eye(K.shape[-1])

        kappa = knn.get_kernel_lebedev(pos_mic, pos_mic, w_pred, dirs, k, integral_weights)
        kappa = kappa.detach().cpu().numpy()

        est_filt = np.moveaxis(np.linalg.solve(np.moveaxis(K_reg, -1, -2), kappa), -1, -2)
        p_nn = np.squeeze(est_filt @ p_mic[:,None])

        mse = round(10*np.log10(np.mean(np.abs(p_nn - p_mic)**2)), num_digits)
        current_loss = round((train_loss.detach().cpu().numpy()).item(), num_digits)
        print(f"epoch: {epochs + 1}\tTrain Loss : {current_loss}\t\tMSE : {mse} dB")
        
        if epochs >= 4 and prev_mse < mse:
            model = prev_model
            break

        prev_model = model
        prev_mse = mse


    # MODEL EVALUATION
    w_pred = model(X)
    K = knn.get_kernel_lebedev(pos_mic, pos_mic, w_pred, dirs, k, integral_weights)
    K = K.detach().cpu().numpy()
    K_reg = K + reg_param * np.eye(K.shape[-1])

    kappa = knn.get_kernel_lebedev(pos_mic, pos_mic, w_pred, dirs, k, integral_weights)
    kappa = kappa.detach().cpu().numpy()

    est_filt_mic = np.moveaxis(np.linalg.solve(np.moveaxis(K_reg, -1, -2), kappa), -1, -2)
    p_nn_mic = np.squeeze(est_filt_mic @ p_mic[:,None])

    kappa_eval = knn.get_kernel_lebedev(pos_mic, pos_eval, w_pred, dirs, k, integral_weights)
    kappa_eval = kappa_eval.detach().cpu().numpy()

    est_filt_eval = np.moveaxis(np.linalg.solve(np.moveaxis(K_reg, -1, -2), kappa_eval), -1, -2)
    p_nn_eval = np.squeeze(est_filt_eval @ p_mic[:,None])

    print(f"MSE kernel interpolation on mic positions: {10*np.log10(np.mean(np.abs(p_nn_mic - p_mic)**2))} dB")
    print(f"MSE kernel interpolation on eval positions: {10*np.log10(np.mean(np.abs(p_nn_eval - p_eval[f_idx,:])**2))} dB")
    # ip.image_single_freq(p_nn_eval, p_eval[f_idx,:], round(freqs[f_idx],2), pos_eval)
    # ip.image_single_freq(p_nn_mic, p_mic, freqs[f_idx], pos_mic)
    # ip.image_single_freq_abs(p_nn, p_mic, freqs[f_idx], pos_mic)
    breakpoint()


def generate_data(sr, num_mic=10):
    rng = np.random.default_rng(10)
    side_len = 1
    #num_mic = 10
    
    #pos_scat = np.array([0, 0, 0])
    sct_region = reg.Disc(0.15, (0.15, -0.15, 0), (0.05, 0.05))
    pos_scat = sct_region.center
    sct_points = sct_region.equally_spaced_points()

    eval_region = reg.Rectangle((side_len, side_len), (0,0,0), (0.05, 0.05))
    pos_eval = eval_region.equally_spaced_points()

    pos_mic = np.zeros((num_mic+1, 3))
    pos_mic[:-1,:2] = rng.uniform(low=-side_len/2, high=side_len/2, size=(num_mic, 2))
    #pos_mic[:,:2] = rng.uniform(low=-side_len/2, high=side_len/2, size=(num_mic, 2))
    pos_mic[-1,:] = pos_scat
    pos_src = np.array([[2,1,-0.6]])
    #pos_src = np.array([pos_scat])

    setup = SimulatorSetup(pathlib.Path(__file__).parent.joinpath("figs"))
    setup.sim_info.samplerate = sr

    setup.sim_info.tot_samples = sr
    setup.sim_info.export_frequency = setup.sim_info.tot_samples
    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [7, 4, 2]
    setup.sim_info.room_center = [2, 0.1, 0.1]
    setup.sim_info.rt60 =  0.1
    setup.sim_info.max_room_ir_length = sr // 2
    setup.sim_info.array_update_freq = 1
    setup.sim_info.randomized_ism = False
    setup.sim_info.auto_save_load = False
    setup.sim_info.sim_buffer = sr // 2
    setup.sim_info.extra_delay = 40
    setup.sim_info.plot_output = "none"

    setup.add_mics("mic", pos_mic)
    setup.add_mics("eval", pos_eval)
    setup.add_controllable_source("src", pos_src)

    sim = setup.create_simulator()

    num_freqs = 128
    freqs = fd.get_real_freqs(num_freqs, sr)
    num_real_freqs = freqs.shape[0]

    fpaths = {}
    for src, mic, path in sim.arrays.iter_paths():
        fpaths.setdefault(src.name, {})
        fpaths[src.name][mic.name] = np.moveaxis(np.moveaxis(np.fft.fft(path, n=num_freqs), -1, 0),1,2)[:num_real_freqs,...]

    return sim.arrays["mic"].pos, fpaths["src"]["mic"][...,0], sim.arrays["eval"].pos, fpaths["src"]["eval"][...,0], freqs, sim.folder_path, sim.sim_info, pos_scat, sct_region, sct_points, side_len, pos_src[0]



if __name__ == "__main__":
    run_exp()
