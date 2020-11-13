from numba import jit
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from scipy.stats import norm, uniform

warnings.filterwarnings('ignore')

sns.set_style("darkgrid")

@jit
def iterate(c, n=100):

    # Iterates complex number n times, or until the iteration shows divergence.
    # Outputs the number of iterations until divergence or 0 for numbers in mandelbrot set.
    if not isinstance(c, complex):
        raise TypeError("Only complex numbers allowed")
    z = 0
    for i in range(n):
        if abs(z) > 2:
            return i
        z = z ** 2 + c
    return 0

@jit
def set_mb(range_real, range_im, size, n=100):
    # Creates set for given ranges, range_real, range_im and size are tuples.
    range_real = np.linspace(range_real[0], range_real[1], size[0])
    range_im = np.linspace(range_im[0], range_im[1], size[1])
    mb_set = np.zeros(size)

    for i in range(size[0]):
        for j in range(size[1]):
            mb_set[i, j] = iterate(complex(range_real[i], range_im[j]), n)
    return mb_set

@jit
def view_mb(range_real, range_im, size, n=100, dpi=100, colormap='Blues_r'):

    # Define colors en plot the mandelbrot set with them
    palette = sns.color_palette(colormap, n)
    delta_real = range_real[1] - range_real[0]
    delta_im = range_im[1] - range_im[0]
    if delta_real >= delta_im:
        x_dim = int(dpi)
        y_dim = int(x_dim * (delta_im / delta_real))
        delta = delta_real / x_dim
    else:
        y_dim = int(dpi)
        x_dim = int(y_dim * (delta_real / delta_im))
        delta = delta_im / dpi

    colors = np.zeros((y_dim, x_dim, 3))
    for i in range(x_dim):
        for j in range(y_dim):
            c = complex(range_real[0] + i * delta, range_im[0] + j * delta)
            n_iter = iterate(c, n)
            if n_iter == 0:
                colors[j, i] = (0, 0, 0)
            else:
                colors[j, i] = palette[n_iter]

    plt.figure(figsize=size, dpi=dpi)
    plt.imshow(colors, zorder=1, interpolation='none')
    ax = plt.gca()
    ax.set_yticks([0, y_dim / 2, y_dim])
    ax.set_yticklabels([range_im[0], (range_im[0]+range_im[1]) / 2, range_im[1]])
    ax.set_xticks([0, x_dim / 2, x_dim])
    ax.set_xticklabels([range_real[0], (range_real[0]+range_real[1]) / 2, range_real[1]])


@jit
def ortho_sample(s, range_real, range_im, gridsize):

    # Produces orthogonal sample of size s within ranges range_real and range_im
    # Gridsize defines number of cut-offs on both axis

    s_sub = int(s / gridsize)
    s_sub2 = int(s / (gridsize ** 2))

    lim_real = np.linspace(range_real[0], range_real[1], (s + 1))
    help_real = np.random.uniform(lim_real[0:s], lim_real[1:(s + 1)])
    lim_im = np.linspace(range_im[0], range_im[1], (s + 1))
    help_im = np.random.uniform(lim_im[0:s], lim_im[1:(s + 1)])

    sample_real = np.empty(s)
    sample_im = np.empty(s)
    for p in range(gridsize):
        sample_real[p * s_sub:(p + 1) * s_sub] = np.random.permutation(help_real[p * s_sub:(p + 1) * s_sub])
        sample_im_t = np.random.permutation(help_im[p * s_sub:(p + 1) * s_sub])
        for q in range(gridsize):
            index = q * s_sub + p * s_sub2
            sample_im[index:index + s_sub2] = sample_im_t[q * s_sub2:(q + 1) * s_sub2]
    return sample_real, sample_im

@jit
def est_area(n_list, s_list, reps=50, range_real=(-2, .5), range_im=(-1.1, 1.1), sampling='pure', gridsize=5):

    total_area = (range_real[1]-range_real[0])*(range_im[1]-range_im[0])
    area = np.empty((len(n_list), len(s_list), reps))
    for n in n_list:
        for s in s_list:
            for i in range(reps):
                if sampling == 'importance':
                    sample_real = np.random.uniform(range_real[0], range_real[1], s)
                    sample_im = np.random.normal(0, 0.5, s)
                    mandel = 0
                    for s_r, s_im in zip(sample_real, sample_im):
                        n_iter = iterate(complex(s_r, s_im), n)
                        if n_iter == 0:
                            phi1 = uniform.pdf(s_im, loc=range_im[0], scale=range_im[1] - range_im[0])
                            phi2 = norm.pdf(s_im, loc=0, scale=0.5)
                            mandel += 1 * phi1 / phi2
                elif sampling == 'importance2':
                    sample_real = np.random.normal(-0.75, 0.6, s)
                    sample_im = np.random.uniform(range_im[0], range_im[1], s)
                    mandel = 0
                    for s_r, s_im in zip(sample_real, sample_im):
                        n_iter = iterate(complex(s_r, s_im), n)
                        if n_iter == 0:
                            phi1 = uniform.pdf(s_r, loc=range_real[0], scale=range_real[1] - range_real[0])
                            phi2 = norm.pdf(s_r, loc=-0.75, scale=0.6)
                            mandel += 1 * phi1 / phi2
                else:
                    if sampling == 'pure':
                        sample_real = np.random.uniform(range_real[0], range_real[1], s)
                        sample_im = np.random.uniform(range_im[0], range_im[1], s)
                    elif sampling == 'lhs':
                        lim_real = np.linspace(range_real[0], range_real[1], (s + 1))
                        sample_real = np.random.permutation(np.random.uniform(lim_real[0:s], lim_real[1:(s + 1)]))
                        lim_im = np.linspace(range_im[0], range_im[1], (s + 1))
                        sample_im = np.random.uniform(lim_im[0:s], lim_im[1:(s + 1)])
                    elif sampling == 'ortho':
                        sample_real, sample_im = ortho_sample(s, range_real, range_im, gridsize)
                    else:
                        print('Undefined sampling method, used pure sampling instead.')
                        sample_real = np.random.uniform(range_real[0], range_real[1], s)
                        sample_im = np.random.uniform(range_im[0], range_im[1], s)
                    mandel = 0
                    for s_r, s_im in zip(sample_real, sample_im):
                        n_iter = iterate(complex(s_r, s_im), n)
                        if n_iter == 0:
                            mandel += 1
                area[n_list.index(n)][s_list.index(s)][i] = mandel/s*total_area
    return area
