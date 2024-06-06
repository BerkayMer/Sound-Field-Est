import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle

from pylebedev import PyLebedev

import torch
import torch.nn as nn

class Kernel_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(3,8)
        self.act1 = nn.Tanh()
        self.hidden2 = nn.Linear(8,8)
        self.act2 = nn.Tanh()
        self.hidden3 = nn.Linear(8,5)
        self.act3 = nn.Tanh()
        self.output = nn.Linear(5,1)

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = torch.exp(self.output(x))
        return x

class EarlyStopping:
    def __init__(self, limit=3):
        self.limit = limit
        self.counter = 0
        self.stop_flag = False

    def __call__(self, prev_mse, mse):
        if mse > prev_mse:
            self.counter += 1
            if self.counter >= self.limit:
                self.stop_flag = True

def dir_vecs_lebedev(order):
    leblib = PyLebedev()

    dirs, weights = leblib.get_points_and_weights(order)

    return dirs, weights

def dir_vecs_fibonacci(num_vecs):
    vecs = []
    phi = np.pi * (np.sqrt(5.)-1.)

    for i in range(num_vecs):
        y = 1 - (i / float(num_vecs - 1)) * 2
        radius = np.sqrt(1 - y*y)

        theta = phi * i

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
    
        vecs.append((x,y,z))

    return np.array(vecs)

def plot_vecs(dir_vecs, dir_vecs2):
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))

    fig, ax = plt.subplots(1, 2, subplot_kw={'projection':'3d', 'aspect':'equal'})
    ax[0].plot_wireframe(x, y, z, color='k', rstride=1, cstride=1, linewidth=0.5)
    ax[0].scatter(dir_vecs[:,0], dir_vecs[:,1], dir_vecs[:,2], s=15, c='r', zorder=10)

    ax[1].plot_wireframe(x, y, z, color='k', rstride=1, cstride=1, linewidth=0.5)
    ax[1].scatter(dir_vecs2[:,0], dir_vecs2[:,1], dir_vecs2[:,2], s=15, c='r', zorder=10)
    plt.show()

def plot_env(pos_mic_ns, pos_mic_s, pos_mic_nn, pos_src, pos_sct):
    x = np.linspace(-0.5,0.5,20)
    y = np.linspace(-0.5,0.5,20)
    z = np.zeros(x.size)

    sct = Circle((pos_sct[0],pos_sct[1]), 0.15, color='k')
    x1, y1 = np.meshgrid(x,y)
    z1, z1 = np.meshgrid(z,z)

    fig, ax = plt.subplots(1, 3, subplot_kw={'projection':'3d', 'aspect':'equal'})
    ax[0].plot_wireframe(x1, y1, z1, color='k', rstride=1, cstride=1, linewidth=0.2)
    ax[0].scatter(pos_mic_ns[:,0], pos_mic_ns[:,1], pos_mic_ns[:,2], s=15, marker='^', c='r', zorder=10)
    ax[0].scatter(pos_src[0], pos_src[1], pos_src[2], s=30, marker='*', c='b', zorder=10)
    ax[0].set_xlim(-0.75,2.25)
    ax[0].set_ylim(-0.75,2.25)
    ax[0].set_xlabel('x(m)')
    ax[0].set_ylabel('y(m)')
    ax[0].set_zlabel('z(m)')

    ax[1].plot_wireframe(x1, y1, z1, color='k', rstride=1, cstride=1, linewidth=0.2)
    ax[1].scatter(pos_mic_s[:,0], pos_mic_s[:,1], pos_mic_s[:,2], s=15, marker='^', c='r', zorder=10)
    ax[1].scatter(pos_src[0], pos_src[1], pos_src[2], s=30, marker='*', c='b', zorder=10)
    ax[1].add_patch(sct)
    art3d.pathpatch_2d_to_3d(sct, z=0, zdir="z")
    ax[1].set_xlim(-0.75,2.25)
    ax[1].set_ylim(-0.75,2.25)
    ax[1].set_xlabel('x(m)')
    ax[1].set_ylabel('y(m)')
    ax[1].set_zlabel('z(m)')

    ax[2].plot_wireframe(x1, y1, z1, color='k', rstride=1, cstride=1, linewidth=0.2)
    ax[2].scatter(pos_mic_nn[:,0], pos_mic_nn[:,1], pos_mic_nn[:,2], s=15, marker='^', c='r', zorder=10)
    ax[2].scatter(pos_src[0], pos_src[1], pos_src[2], s=30, marker='*', c='b', zorder=10)
    ax[2].set_xlim(-0.75,2.25)
    ax[2].set_ylim(-0.75,2.25)
    ax[2].set_xlabel('x(m)')
    ax[2].set_ylabel('y(m)')
    ax[2].set_zlabel('z(m)')

    plt.show()

def get_kernel(points1, points2, weights, dir_vecs, k):
    dist_mat = np.expand_dims(points1,1) - np.expand_dims(points2,0)
    exponential = np.exp(1j * k * np.dot(dist_mat, dir_vecs.T))
    exponential = torch.tensor(exponential, dtype=torch.cfloat)
    kernel = torch.squeeze(torch.unsqueeze(exponential,-1) * weights)

    # integrate over dir_vecs
    K = torch.trapezoid(kernel)
    return K

def get_kernel_lebedev(points1, points2, weights, dir_vecs, k, int_weights):
    int_weights = torch.tensor(int_weights, dtype=torch.float32)
    sphere_area = 4 * np.pi
    
    '''
    # weight normalization
    integral = sphere_area * torch.sum(torch.unsqueeze(int_weights,-1) * weights, axis=0)
    weights = weights / integral
    '''

    dist_mat = np.expand_dims(points1,1) - np.expand_dims(points2,0)
    exponential = np.exp(-1j * k * np.dot(dist_mat, dir_vecs.T))
    exponential = torch.tensor(exponential, dtype=torch.cfloat)
    kernel = torch.squeeze(torch.unsqueeze(exponential,-1) * weights)
    
    K = sphere_area * torch.sum(int_weights * kernel, axis=2)

    return K

def loss_fn(K, s, reg_param):
    '''
        Implements the loss function defined in equation (51)
        described in 'Sound Field Estimation Based on
        Physics-Constrained Kernel Interpolation
        Adapted to Environment' by Ribiero et. al. 2023
    '''
    s = torch.unsqueeze(s,-1)
    loss = torch.real(reg_param * torch.transpose(s.conj(),0,1) @ torch.linalg.solve(K + reg_param * torch.eye(K.shape[0], dtype=torch.cfloat), s))
    
    return torch.squeeze(loss)

def tensor_round(input, num_digits):
    return torch.round(input * 10**num_digits) / (10**num_digits)