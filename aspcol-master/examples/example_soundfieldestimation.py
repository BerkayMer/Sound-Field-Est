import matplotlib.pyplot as plt
import scipy.signal as ss
import numpy as np
import pathlib


from aspsim.simulator import SimulatorSetup
from aspsim.room      import roomimpulseresponse as rirs
import aspsim.room.region as reg

import aspcol.kernelinterpolation as ki
import aspcol.scatterki as scki
import aspcol.filterdesign as fd
import aspcol.soundfieldestimation as sfe
import aspcol.ir_freefield as irfr
import aspcol.irplot as ip


def run_exp():
    samplerate = 1000
    pos_mic, p_mic, pos_eval, p_eval, freqs, fig_folder, sim_info, pos_sct, sct_region, sct_points, side_len, pos_src = generate_data(samplerate)
    pos_sct = pos_sct.reshape(1,-1)
    wave_num = 2 * np.pi * freqs / sim_info.c
    min_dly = 0
    extra_delay = 20
    frac_dly_len = 2*(min_dly+extra_delay)+1

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
    #p_eval_new[:,sct_region.is_in_region(pos_eval)] = np.zeros((freqs.shape[0], p_eval_new[:,sct_region.is_in_region(pos_eval)].shape[1]))

    pos_mic = np.delete(pos_mic, sct_region.is_in_region(pos_mic), axis=0)
    pos_eval = np.delete(pos_eval, sct_region.is_in_region(pos_eval), axis=0)

    p_mic  = p_mic[:, np.any(p_mic, axis=0)]
    p_eval = p_eval[:, np.any(p_eval, axis=0)]
    p_eval_inc = p_eval_inc[:, np.any(p_eval_inc, axis=0)]
    p_sct_eval_freq = p_sct_eval_freq[:, np.any(p_sct_eval_freq, axis=0)]
    #p_eval_new = p_eval_new[:, np.any(p_eval, axis=0)]

    #breakpoint()
    
    reg_param = 1e-5
    reg_param2 = 1e-8

    N = (np.ceil(R*wave_num)).astype(int)

    est_nni = scki.nearest_neighbor(pos_mic, pos_eval, p_mic)
    est_ki = sfe.est_ki_diffuse_freq(p_mic, pos_mic, pos_eval, wave_num, reg_param)
    est_ki_dir = est_ki_directional_freq(p_mic, pos_mic, pos_eval, wave_num, reg_param, np.array([[1,0,0]]), 5)
    est_ki_sct = est_ki_scatter_freq_varN(p_mic, pos_mic, pos_eval, pos_sct, sct_region, wave_num, reg_param, reg_param2, N)
    est_ki_sct_2N = est_ki_scatter_freq_varN(p_mic, pos_mic, pos_eval, pos_sct, sct_region, wave_num, reg_param, reg_param2, 2*N)
    breakpoint()

    print(f"MSE nearest neighbor interpolation: {10*np.log10(np.mean(np.abs(est_nni - p_eval)**2))} dB")
    print(f"MSE kernel interpolation: {10*np.log10(np.mean(np.abs(est_ki - p_eval)**2))} dB")
    print(f"MSE directional kernel interpolation: {10*np.log10(np.mean(np.abs(est_ki_dir - p_eval)**2))} dB")
    print(f"MSE scattering kernel interpolation [N]: {10*np.log10(np.mean(np.abs(est_ki_sct - p_eval[1:])**2))} dB")
    print(f"MSE scattering kernel interpolation [2N]: {10*np.log10(np.mean(np.abs(est_ki_sct_2N - p_eval[1:])**2))} dB")

    fig, ax = plt.subplots(1,1)

    ax.plot(freqs[15:], 10*np.log10(np.mean(np.abs(est_nni[15:] - p_eval[15:])**2, axis=-1)), label="nearest neigbor")
    ax.plot(freqs[15:], 10*np.log10(np.mean(np.abs(est_ki[15:] - p_eval[15:])**2, axis=-1)), label="diffuse KI")
    ax.plot(freqs[15:], 10*np.log10(np.mean(np.abs(est_ki_dir[15:] - p_eval[15:])**2, axis=-1)), label="directional KI")
    ax.plot(freqs[15:], 10*np.log10(np.mean(np.abs(est_ki_sct[14:] - p_eval[15:])**2, axis=-1)), label="scattering KI [kR]")
    ax.plot(freqs[15:], 10*np.log10(np.mean(np.abs(est_ki_sct_2N[14:] - p_eval[15:])**2, axis=-1)), label="scattering KI 2[kR]")

    #ax.plot(freqs[1:], 10*np.log10(np.mean(np.abs(est_nni[1:] - p_eval[1:])**2, axis=-1)), label="nearest neigbor")
    #ax.plot(freqs[1:], 10*np.log10(np.mean(np.abs(est_ki[1:] - p_eval[1:])**2, axis=-1)), label="diffuse KI")
    #ax.plot(freqs[1:], 10*np.log10(np.mean(np.abs(est_ki_dir[1:]- p_eval[1:])**2, axis=-1)), label="directional KI")

    ax.set_title(f"")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("MSE [dB]")
    plt.legend()
    plt.grid()
    plt.show()

    #ip.image_scatter_freq_response(p_eval, est_ki_sct, freqs, pos_eval, est_name="scat_est", plot_name="Estimation vs Actual Pressures")
    #ip.image_scatter_freq_response(est_ki_sct, p_eval[1:,:], freqs[1:], pos_eval, est_name="scat_est", plot_name="Estimation vs Actual Pressures")
    
    #ip.image_single_freq(p_mic[102,:], p_mic[102,:], freqs[102], pos_mic)
    #ip.image_single_freq(est_nni[16,:], p_eval[16,:], freqs[16], pos_eval)
    #ip.image_single_freq(est_ki_sct[101,:], p_eval[102,:], round(freqs[102],2), pos_eval)
    
    breakpoint()


def est_ki_directional_freq(p_freq, pos_mic, pos_eval, wave_num, reg_param, direction, direction_param):
    est_filt = ki.get_krr_parameters(ki.kernel_directional_3d, reg_param, pos_eval, pos_mic, wave_num, direction, direction_param)[:,0,:,:]
    p_ki = est_filt @ p_freq[:,:,None]
    return np.squeeze(p_ki, axis=-1)

def est_ki_scatter_freq(p_freq, pos, pos_eval, pos_sct, k, reg_param1, reg_param2, N):
    eval_expand, sct_filt, K_reg, sph_coeffs, kappa = scki.get_params_sct(ki.kernel_helmholtz_3d, reg_param1, reg_param2, pos_eval, pos, pos_sct, N, k) # for NN consider directional kernel
    inc_params = np.moveaxis(np.linalg.solve(np.moveaxis(K_reg, -1, -2), kappa), -1, -2)
    
    temp = (p_freq[1:] - np.squeeze(sph_coeffs @ sct_filt @ p_freq[1:,:,None]))
    p_inc = inc_params @ temp[:,:,None]

    p_sct = eval_expand @ sct_filt @ p_freq[1:,:,None]
    p_eval = p_inc + p_sct

    return np.squeeze(p_eval, axis=-1)

def est_ki_scatter_freq_varN(p_freq, pos, pos_eval, pos_sct, sct_region, k, reg_param1, reg_param2, N):
    eval_expand, sct_filt, K_reg, sph_coeffs, kappa = scki.get_params_sct(ki.kernel_helmholtz_3d, reg_param1, reg_param2, pos_eval, pos, pos_sct, sct_region, N, k)
    inc_params = np.moveaxis(np.linalg.solve(np.moveaxis(K_reg, -1, -2), kappa), -1, -2)
    p_eval = np.zeros((len(k)-1,len(pos_eval),1), dtype=complex)
    #p_eval_inc = np.zeros((len(k)-1,len(pos_eval),1), dtype=complex)

    for idx in range(len(k)-1):
        eval_expand[idx][sct_region.is_in_region(pos_eval),:] = 0
        temp = (p_freq[idx+1] - np.squeeze(sph_coeffs[idx] @ sct_filt[idx] @ p_freq[idx+1,:,None]))
        p_inc = inc_params[idx,:,:] @ temp[:,None]
        p_sct = eval_expand[idx] @ sct_filt[idx] @ p_freq[idx+1,:,None]
        p_eval[idx] = p_inc + p_sct
        #p_eval[idx] = p_inc

    return np.squeeze(p_eval, axis=-1)


def generate_data(sr):
    rng = np.random.default_rng(10)
    side_len = 1
    num_mic = 40
    
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
    pos_src = np.array([[2,2,-0.6]])
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