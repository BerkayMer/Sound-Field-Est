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
    pos_mic, p_mic, pos_eval, p_eval, freqs, fig_folder, sim_info, pos_src = generate_data(samplerate)
    wave_num = 2 * np.pi * freqs / sim_info.c
    src_vector = pos_src/np.linalg.norm(pos_src)
    reg_param = 1e-8

    est_nni = scki.nearest_neighbor(pos_mic, pos_eval, p_mic)
    est_ki = sfe.est_ki_diffuse_freq(p_mic, pos_mic, pos_eval, wave_num, reg_param)
    est_ki_dir = est_ki_directional_freq(p_mic, pos_mic, pos_eval, wave_num, reg_param, src_vector, 2)

    print(f"MSE nearest neighbor interpolation: {10*np.log10(np.mean(np.abs(est_nni - p_eval)**2))} dB")
    print(f"MSE kernel interpolation: {10*np.log10(np.mean(np.abs(est_ki - p_eval)**2))} dB")
    print(f"MSE directional kernel interpolation: {10*np.log10(np.mean(np.abs(est_ki_dir - p_eval)**2))} dB")

    fig, ax = plt.subplots(1,1)
    ax.plot(freqs[1:], 10*np.log10(np.mean(np.abs(est_nni[1:] - p_eval[1:])**2, axis=-1)), label="nearest neigbor")
    ax.plot(freqs[1:], 10*np.log10(np.mean(np.abs(est_ki[1:] - p_eval[1:])**2, axis=-1)), label="diffuse KI")
    ax.plot(freqs[1:], 10*np.log10(np.mean(np.abs(est_ki_dir[1:] - p_eval[1:])**2, axis=-1)), label="directional KI")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("MSE [dB]")
    plt.legend()
    plt.grid()
    plt.show()

    #ip.image_single_freq(p_mic[102,:], p_mic[102,:], freqs[102], pos_mic)
    #ip.image_single_freq(est_nni[16,:], p_eval[16,:], freqs[16], pos_eval)
    #ip.image_single_freq(est_ki[101,:], p_eval[102,:], freqs[102], pos_eval)
    #ip.soundfield_plot(est_nni[60,:],est_ki[60,:],est_ki_dir[60,:],p_eval[60,:],pos_eval) 

    breakpoint()

def est_ki_directional_freq(p_freq, pos_mic, pos_eval, wave_num, reg_param, direction, direction_param):
    est_filt = ki.get_krr_parameters(ki.kernel_directional_3d, reg_param, pos_eval, pos_mic, wave_num, direction, direction_param)[:,0,:,:]
    p_ki = est_filt @ p_freq[:,:,None]
    return np.squeeze(p_ki, axis=-1)




def generate_data(sr):
    rng = np.random.default_rng(10)
    side_len = 1
    num_mic = 10
    
    eval_region = reg.Rectangle((side_len, side_len), (0,0,0), (0.05, 0.05))
    pos_eval = eval_region.equally_spaced_points()

    pos_mic = np.zeros((num_mic, 3))
    pos_mic[:,:2] = rng.uniform(low=-side_len/2, high=side_len/2, size=(num_mic, 2))
    pos_src = np.array([[2,1,-0.6]])

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

    return sim.arrays["mic"].pos, fpaths["src"]["mic"][...,0], sim.arrays["eval"].pos, fpaths["src"]["eval"][...,0], freqs, sim.folder_path, sim.sim_info, sim.arrays["src"].pos



if __name__ == "__main__":
    run_exp()