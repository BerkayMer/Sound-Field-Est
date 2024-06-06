import numpy as np
import scipy.spatial.distance as distfuncs
import scipy.special as special
import numba as nb

import aspcol.utilities as util
import aspcol.filterdesign as fd
import aspcol.montecarlo as mc
import aspsim.room.region as reg

import aspcol.kernelinterpolation as ki

"""
    This files implements the required functions to design the approach described in
    'Kernel Interpolation of Incident Sound Field in Region
    Including Scattering Objects' by Koyama et al. 2023
"""


def nearest_neighbor(points1, points2, p_mic):
    dist_mat = np.expand_dims(points1,1) - np.expand_dims(points2,0)
    dist_mat = np.linalg.norm(dist_mat, axis=2)
    nearest_idx = np.argmin(dist_mat, axis=0)

    return p_mic[:,nearest_idx]

def spherical_exp(points, exp_center, sct_region, wavenum, N):
    num_mics = points.shape[0]

    points = points - exp_center
    dist = np.linalg.norm(points, axis=1)
    plane_dist = np.linalg.norm(points[:,:2], axis=1)

    theta = np.arctan2(points[:,1],points[:,0])
    phi = np.arctan2(plane_dist, points[:,2])

    freqs_coeffs = [None] * (wavenum.shape[0]-1)
    for idx in range(wavenum.shape[0]-1):
        exp_coeffs = []
        N_freq = N[idx+1]
        
        for mic in range(num_mics):
            mic_coeffs = []
            
            for n in range(N_freq+1):
                for m in range(-1*n, n+1):
                    coeff = np.sqrt(4*np.pi)*(special.spherical_jn(n,wavenum[idx+1]*dist[mic])-1j*special.spherical_yn(n,wavenum[idx+1]*dist[mic]))*special.sph_harm(m,n,theta[mic],phi[mic])
                    mic_coeffs = np.append(mic_coeffs, coeff)
            
            if mic == 0:
                exp_coeffs = np.concatenate((exp_coeffs, mic_coeffs))
            else:
                exp_coeffs = np.vstack((exp_coeffs, mic_coeffs))
        
        freqs_coeffs[idx] = exp_coeffs #return this as a list for different N

    return freqs_coeffs, dist, theta, phi


def sph_weights(dist, theta, phi, wavenum, N):
    num_mics = dist.shape[0]
    W_freqs = [None] * (wavenum.shape[0]-1)

    for idx in range(wavenum.shape[0]-1):
        theta_der = []
        phi_der = []
        N_freq = N[idx+1]
        
        for mic in range(num_mics):
            weight_theta = []
            weight_phi = []
            
            for n in range(N_freq+1):
                for m in range(-1*n, n+1):
                    Y = special.sph_harm(m,n,theta[mic],phi[mic])
                    if np.isnan(Y).any():
                        breakpoint()
                    h = (special.spherical_jn(n,wavenum[idx+1]*dist[mic])-1j*special.spherical_yn(n,wavenum[idx+1]*dist[mic]))
                    
                    phi_dern = np.sqrt(4*np.pi)*1j*m*h*Y

                    if(m == n):
                        theta_dern = np.sqrt(4*np.pi)*h*m*(1/np.tan(theta[mic]))*Y
                        
                    else:
                        add_Y = special.sph_harm(m+1,n,theta[mic],phi[mic])
                        add_term = np.sqrt((n-m)*(n+m+1))*np.exp(-1*1j*phi[mic])*add_Y
                        theta_dern = np.sqrt(4*np.pi)*h*(m*(1/np.tan(theta[mic]))*Y+add_term)
                    
                    weight_theta = np.append(weight_theta, theta_dern)
                    weight_phi = np.append(weight_phi, phi_dern)
                    
            if mic == 0:
                theta_der = np.concatenate((theta_der, weight_theta))
                phi_der = np.concatenate((phi_der, weight_phi))
            else:
                theta_der = np.vstack((theta_der, weight_theta))
                phi_der = np.vstack((phi_der, weight_phi))

        theta_der = np.matrix(theta_der)
        phi_der = np.matrix(phi_der)

        W_freqs[idx] = np.array(theta_der.H @ theta_der + phi_der.H @ phi_der)


    return W_freqs  #return as a list if varying N


def get_params_sct(kernel_func, reg_param1, reg_param2, output_arg, data_arg, pos_sct, sct_region, N, *args):
    """
    Calculates the parameter matrix for KRR together with scattering object field
    As described in 'Kernel Interpolation of Incident Sound Field in Region
    Including Scattering Objects' by Koyama et al.
    """
    # KRR side
    K = kernel_func(data_arg, data_arg, *args)
    K_reg = K + reg_param1 * np.eye(K.shape[-1])

    # Spherical wave expansion side
    sph_coeffs, dist, theta, phi = spherical_exp(data_arg, pos_sct, sct_region, *args, N)
    eval_coeffs = spherical_exp(output_arg, pos_sct, sct_region, *args, N)[0]
    
    
    # weighting matrix
    W_freqs = sph_weights(dist, theta, phi, *args, N)
    soln_u = [None] * len(W_freqs)

    for idx in range(len(W_freqs)):
        N_freq = N[idx+1]
        coeff_mat = np.matrix(sph_coeffs[idx])
        W_mat = np.matrix(W_freqs[idx])

        u_reg_part = coeff_mat.H @ np.linalg.inv(K_reg[idx+1]) @ coeff_mat
        u_reg_part = np.linalg.inv(u_reg_part + (reg_param2/reg_param1)*W_mat)
        soln_u[idx] = np.array(u_reg_part @ coeff_mat.H @ np.linalg.inv(K_reg[idx+1]))
    
    sct_params = soln_u

    kappa = np.moveaxis(kernel_func(output_arg, data_arg, *args), -1, -2)
    kappa = kappa[1:]

    return  eval_coeffs, sct_params, K_reg[1:], sph_coeffs, kappa