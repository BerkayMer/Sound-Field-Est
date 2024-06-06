import matplotlib.pyplot as plt
import numpy as np
import scipy.special as special


#def image_scatter_freq_response(ir_all_freq, actual_ir, freqs, pos, fig_folder, est_name="", plot_name=""):
def image_scatter_freq_response(ir_all_freq, actual_ir, freqs, pos, est_name="", plot_name=""):
    """
    ir_all_freq is a dict, where each value is a ndarray of shape (num_freq, num_pos)
    freqs is a 1-d np.ndarray with all frequencies
    pos is a ndarray of shape (num_pos, 3)
    """
    num_freqs = freqs.shape[-1]
    num_example_freqs = 4
    idx_interval = num_freqs // (num_example_freqs+1)
    freq_idxs = np.arange(num_freqs)[idx_interval::idx_interval]



    for fi in freq_idxs:
        fig, axes = plt.subplots(1, 2, figsize=(15, len(ir_all_freq)))
        """
        for ax_row, (ir_val) in zip(axes, ir_all_freq):
            #mse_val += 1e-6
            #mse_val = 10 * np.log10(mse_val)

            print(axes.shape)
            breakpoint()
            clr = ax_row[0].scatter(pos[:,0], pos[:,1], c=np.real(ir_val), s=500)
            cbar = fig.colorbar(clr, ax=ax_row[0])
            cbar.set_label('Real pressure')

            clr = ax_row[1].scatter(pos[:,0], pos[:,1], c=np.imag(ir_val), s=500)
            cbar = fig.colorbar(clr, ax=ax_row[1])
            cbar.set_label('Imag pressure')

            clr = ax_row[2].scatter(pos[:,0], pos[:,1], c=np.abs(ir_val), s=500)
            cbar = fig.colorbar(clr, ax=ax_row[2])
            cbar.set_label('Abs pressure')

            ax_row[0].set_title(f"Real: {est_name}")
            ax_row[1].set_title(f"Imag: {est_name}")
            ax_row[2].set_title(f"Abs: {est_name}")

            #for ax in ax_row:
             #   if "moving_mic" in pos:
              #      moving_mic.plot_moving_mic(pos["moving_mic"], ax)
               #     ax.legend(loc="lower left")
               # #ax.legend(loc="lower left")
               # ax.set_xlabel("x (m)")
               # ax.set_ylabel("y (m)")
                
               # ax.set_aspect("equal")
               # #aspplot.set_basic_plot_look(ax)
        """

        clr = axes[0].scatter(pos[:,0], pos[:,1], c=np.real(ir_all_freq[fi,:]), s=500)
        cbar = fig.colorbar(clr, ax=axes[0])
        cbar.set_label('Real pressure')

        clr = axes[1].scatter(pos[:,0], pos[:,1], c=np.real(actual_ir[fi,:]), s=500)
        cbar = fig.colorbar(clr, ax=axes[1])
        cbar.set_label('Real pressure')

        #clr = axes[2].scatter(pos[:,0], pos[:,1], c=np.abs(ir_all_freq[fi,:]), s=500)
        #cbar = fig.colorbar(clr, ax=axes[2])
        #cbar.set_label('Abs pressure')

        axes[0].set_title(f"Real: {est_name}  f: {np.round(freqs[fi],2)}Hz")
        axes[1].set_title(f"Real: p_actual f: {np.round(freqs[fi],2)}Hz")
        #axes[2].set_title(f"Abs: {est_name}")

        plt.show()   
        #aspplot.output_plot("pdf", fig_folder, f"image_scatter_freq_{freqs[fi]}Hz{plot_name}")

def image_single_freq(p_eval, p_actual, f, pos, est_name="", plot_name=""):
    fig, axes = plt.subplots(1, 2, figsize=(15, len(p_eval)))

    clr = axes[0].scatter(pos[:,0], pos[:,1], c=np.real(p_eval), s=500)
    cbar = fig.colorbar(clr, ax=axes[0])
    cbar.set_label('Real pressure')

    clr = axes[1].scatter(pos[:,0], pos[:,1], c=np.real(p_actual), s=500)
    cbar = fig.colorbar(clr, ax=axes[1])
    cbar.set_label('Real pressure')

    axes[0].set_title(f"Neural Network  f: 350 Hz")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[1].set_title(f"Actual f: 350 Hz")
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Y (m)")

    plt.show() 

def image_single_freq_abs(p_eval, p_actual, f, pos, est_name="", plot_name=""):
    fig, axes = plt.subplots(1, 2, figsize=(15, len(p_eval)))

    clr = axes[0].scatter(pos[:,0], pos[:,1], c=np.abs(p_eval), s=500)
    cbar = fig.colorbar(clr, ax=axes[0])
    cbar.set_label('Abs pressure')

    clr = axes[1].scatter(pos[:,0], pos[:,1], c=np.abs(p_actual), s=500)
    cbar = fig.colorbar(clr, ax=axes[1])
    cbar.set_label('Abs pressure')

    axes[0].set_title(f"Abs: p_eval  f: {np.round(f,2)}Hz")
    axes[1].set_title(f"Abs: p_actual f: {np.round(f,2)}Hz")

    plt.show()

def IR_time_plot(src2pnt, sct2pnt, src2sct, res_ir, src_pos, sct_pos, pnt_pos, samplerate, c):
    
    dist = np.sqrt((src_pos[0]-pnt_pos[0])**2 + (src_pos[1]-pnt_pos[1])**2 + (src_pos[2]-pnt_pos[2])**2)

    prop_delay = int((dist/c)*samplerate)
    print(f"Expected Src2pnt Propagation delay(samples): {prop_delay}")
    ir_delay = np.argmax(src2pnt)
    print(f"Delay of the Src2pnt IR: {ir_delay}")

    dist = np.sqrt((sct_pos[0]-pnt_pos[0])**2 + (sct_pos[1]-pnt_pos[1])**2 + (sct_pos[2]-pnt_pos[2])**2)
    prop_delay = int(round((dist/c)*samplerate))
    print(f"Expected Sct2pnt Propagation delay(samples): {prop_delay}")
    ir_delay = np.argmax(sct2pnt)
    print(f"Delay of the Sct2pnt IR: {ir_delay}")

    dist = np.sqrt((sct_pos[0]-src_pos[0])**2 + (sct_pos[1]-src_pos[1])**2 + (sct_pos[2]-src_pos[2])**2)
    prop_delay = int(round((dist/c)*samplerate))
    print(f"Expected Src2sct Propagation delay(samples): {prop_delay}")
    ir_delay = np.argmax(src2sct)
    print(f"Delay of the Src2sct IR: {ir_delay}")

    ir_delay = np.argmax(res_ir)
    print(f"Delay of src2mic*sct2pnt: {ir_delay}")

    fig = plt.figure()
    
    gs  = fig.add_gridspec(2,2)
    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[0,1])
    ax2 = fig.add_subplot(gs[1,:])

    samples = np.array(range(src2pnt.shape[0]))
    ax0.plot(samples, src2pnt)
    ax0.set_title("IR Source to Point")

    samples = np.array(range(sct2pnt.shape[0]))
    ax1.plot(samples, sct2pnt)
    ax1.set_title("IR Scatterrer to Point")

    samples = np.array(range(res_ir.shape[0]))
    ax2.plot(samples, res_ir)
    ax2.set_title("Resulting IR")

    plt.show()
def plot_sph_bessel():
    x = np.linspace(0,5,500)

    j0_x = special.spherical_jn(0,x)
    j1_x = special.spherical_jn(1,x)
    j2_x = special.spherical_jn(2,x)
    j3_x = special.spherical_jn(3,x)
    j4_x = special.spherical_jn(4,x)
    j5_x = special.spherical_jn(5,x)

    n0_x = special.spherical_yn(0,x)
    n1_x = special.spherical_yn(1,x)
    n2_x = special.spherical_yn(2,x)
    n3_x = special.spherical_yn(3,x)
    n4_x = special.spherical_yn(4,x)
    n5_x = special.spherical_yn(5,x)

    fig, axes = plt.subplots(1, 2)
    axes[0].plot(x,j0_x,label="j_0(x)")
    axes[0].plot(x,j1_x,label="j_1(x)")
    axes[0].plot(x,j2_x,label="j_2(x)")
    axes[0].plot(x,j3_x,label="j_3(x)")
    axes[0].plot(x,j4_x,label="j_4(x)")
    axes[0].plot(x,j5_x,label="j_5(x)")
    axes[0].legend()
    axes[0].set_ylim(-0.3,1.3)
    axes[0].set_title(f"j_n(x)")
    axes[0].grid()

    axes[1].plot(x,n0_x,label="n_0(x)")
    axes[1].plot(x,n1_x,label="n_1(x)")
    axes[1].plot(x,n2_x,label="n_2(x)")
    axes[1].plot(x,n3_x,label="n_3(x)")
    axes[1].plot(x,n4_x,label="n_4(x)")
    axes[1].plot(x,n5_x,label="n_5(x)")
    axes[1].legend()
    axes[1].set_ylim(-5,1)
    axes[1].set_title(f"n_n(x)")
    axes[1].grid()

    plt.show()

def plot_field_decomp():
    f = 20         # frequency
    f2 = 30
    f3 = 10
    f4 = 40
    fs = 100       # sample frequency
    Ts = 1/fs      # sample period
    t = np.arange(0,0.5, Ts)   # time index
    c = 50             # speed of wave
    w = 2*np.pi *f     # angular frequency
    w2 = 2*np.pi * f2
    w3 = 2*np.pi * f3
    w4 = 2*np.pi * f4
    w5 = 2*np.pi * f4
    k = w/c            # wave number

    resolution = 0.02
    x = np.arange(-5, 5, resolution)
    y = np.arange(-5, 5, resolution)
    pos_src = np.array([8,-10])
    sct_pos = np.array([-1.5,2])
    sct_pos2 = np.array([0,0])
    sct_pos3 = np.array([4,-2])
    sct_pos4 = np.array([3,3])
    sct_pos5 = np.array([-2,-4])
    dx = np.array(x); M = len(dx)
    dy = np.array(y); N = len(dy)
    [xx, yy] = np.meshgrid(x, y)
    theta = np.pi / 4         # direction of propagation
    kx =  k* np.cos(theta)
    ky = k * np.sin(theta)

    plane_wave = np.sin(np.sqrt(kx * (xx-pos_src[0])**2 + ky * (yy-pos_src[1])**2) - w * t[1])
    sph_wave = np.sin(np.sqrt(kx * (xx-sct_pos[0])**2 + ky * (yy-sct_pos[1])**2) - w * t[1])
    sph_wave2 = np.sin(np.sqrt(kx * (xx-sct_pos2[0])**2 + ky * (yy-sct_pos2[1])**2) - w4 * t[1])
    sph_wave3 = np.sin(np.sqrt(kx * (xx-sct_pos3[0])**2 + ky * (yy-sct_pos3[1])**2) - w4 * t[1])
    sph_wave4 = np.sin(np.sqrt(kx * (xx-sct_pos4[0])**2 + ky * (yy-sct_pos4[1])**2) - w4 * t[1])
    sph_wave5 = np.sin(np.sqrt(kx * (xx-sct_pos5[0])**2 + ky * (yy-sct_pos5[1])**2) - w4 * t[1])
    res_field = sph_wave + sph_wave2 + sph_wave3 + sph_wave4 + sph_wave5


    fig,ax = plt.subplots(1,3, figsize=(15, 5))
    ax[0].pcolor(x,y,plane_wave,cmap='seismic')
    ax[0].set_xlabel("x (m)")
    ax[0].set_ylabel("y (m)")
    ax[0].set_title(f"Incident Field")

    ax[1].pcolor(x,y,sph_wave,cmap='seismic')
    ax[1].set_xlabel("x (m)")
    ax[1].set_ylabel("y (m)")
    ax[1].set_title(f"Scattering Field")

    ax[2].pcolor(x,y,res_field,cmap='seismic')
    ax[2].set_xlabel("x (m)")
    ax[2].set_ylabel("y (m)")
    ax[2].set_title(f"Residual Field")

    plt.show()

    

def soundfield_plot(field1, field2, field3, actual, pos, f):
    fig, axes = plt.subplots(2, 2, figsize=(15, len(field1)))

    clr = axes[0,0].scatter(pos[:,0], pos[:,1], c=np.real(actual), s=500)
    cbar = fig.colorbar(clr, ax=axes[0,0])
    cbar.set_label('Real pressure')

    clr = axes[0,1].scatter(pos[:,0], pos[:,1], c=np.real(field1), s=500)
    cbar = fig.colorbar(clr, ax=axes[0,1])
    cbar.set_label('Real pressure')

    clr = axes[1,0].scatter(pos[:,0], pos[:,1], c=np.real(field2), s=500)
    cbar = fig.colorbar(clr, ax=axes[1,0])
    cbar.set_label('Real pressure')

    clr = axes[1,1].scatter(pos[:,0], pos[:,1], c=np.real(field3), s=500)
    cbar = fig.colorbar(clr, ax=axes[1,1])
    cbar.set_label('Real pressure')
    
    axes[0,0].set_title(f"Actual")
    axes[0,0].set_xlabel(f"X (m)")
    axes[0,0].set_ylabel(f"Y (m)")
    axes[0,1].set_title(f"Nearest Neighbor")
    axes[0,1].set_xlabel(f"X (m)")
    axes[0,1].set_ylabel(f"Y (m)")
    axes[1,0].set_title(f"Diffuse Kernel")
    axes[1,0].set_xlabel(f"X (m)")
    axes[1,0].set_ylabel(f"Y (m)")
    axes[1,1].set_title(f"Directional Kernel")
    axes[1,1].set_xlabel(f"X (m)")
    axes[1,1].set_ylabel(f"Y (m)")
    
    fig.suptitle(f'Estimates for f: {f} Hz')

    plt.show()
