import scipy.spatial.distance as distfuncs
import numpy as np

def ir_point_source_3d(pos_from, pos_to, samplerate, c, max_order=25):
    dist = distfuncs.cdist(pos_from, pos_to)
    sample_delay = (samplerate * dist) / c
    max_filt_len = int(np.max(sample_delay)) + 1 + (max_order // 2) + 1
    ir = np.zeros((dist.shape[0], dist.shape[1], max_filt_len))

    attenuation = lambda d: 1 / (4 * np.pi * np.max((d, 1e-2)))

    for row in range(dist.shape[0]):
        for col in range(dist.shape[1]):
            ir[row, col, :] = _frac_delay_lagrange_filter(
                sample_delay[row, col], max_order, max_filt_len
            )
            ir[row, col, :] *= attenuation(dist[row, col])
    return ir
	
	
def _frac_delay_lagrange_filter(delay, max_order, max_filt_len):
    """Generates fractional delay filter using lagrange interpolation
    if maxOrder would require a longer filter than maxFiltLen,
    maxFiltLen will will override and set a lower interpolation order"""
    ir = np.zeros(max_filt_len)
    frac_dly, dly = np.modf(delay)
    order = int(np.min((max_order, 2 * dly, 2 * (max_filt_len - 1 - dly))))

    delta = np.floor(order / 2) + frac_dly
    h = _lagrange_interpol(order, delta)

    diff = delay - delta
    start_idx = int(np.floor(diff))
    ir[start_idx : start_idx + order + 1] = h
    return ir


def _lagrange_interpol(N, delta):
    ir_len = int(N + 1)
    h = np.zeros(ir_len)

    for n in range(ir_len):
        k = np.arange(ir_len)
        k = k[np.arange(ir_len) != n]

        h[n] = np.prod((delta - k) / (n - k))
    return h