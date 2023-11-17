import numpy as np
import scipy.spatial.distance as distfuncs
import scipy.special as special
import numba as nb
import matplotlib.pyplot as plt


import functions as func

fs = 8000
c = 343 # in m/s
reg_param = 1e-3

pos_mic, p_mic, pos_eval, p_eval, freqs = func.generate_pressure_data(fs)
wave_num = freqs / c
print(np.shape(pos_eval))
est_ki_diff = func.diffuse_est(p_mic,pos_mic,pos_eval,wave_num,reg_param)
est_ki_dir  = func.directional_est(p_mic,pos_mic,pos_eval,wave_num,reg_param,np.array([[1,0,0]]),5)

print(f"MSE of diffuse interpolation: {10*np.log10(np.mean(np.abs(est_ki_diff - p_eval)**2))} dB")
print(f"MSE of directional interpolation: {10*np.log10(np.mean(np.abs(est_ki_dir - p_eval)**2))} dB")

fig, ax = plt.subplots(1,1)
ax.plot(freqs, 10*np.log10(np.mean(np.abs(est_ki_diff - p_eval)**2, axis=-1)), label="diffuse KI")
ax.plot(freqs, 10*np.log10(np.mean(np.abs(est_ki_dir - p_eval)**2, axis=-1)), label="directional KI")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("MSE [dB]")
plt.legend()
plt.show()





    