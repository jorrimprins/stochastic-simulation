from numba import jit
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
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
    ## Creates set for given ranges, range_real, range_im and size are tuples.
    range_real = np.linspace(range_real[0],range_real[1],size[0])
    range_im = np.linspace(range_im[0],range_im[1],size[1])
    mb_set = np.zeros(size)

    for i in range(size[0]):
        for j in range(size[1]):
            mb_set[i, j] = iterate(complex(range_real[i], range_im[j]), n)
    return mb_set

@jit
def view_mb(range_real, range_im, size, n=100, dpi=100, colormap='Blues_r'):

    #Define colors
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
            iter = iterate(c, n)
            if iter == 0:
                colors[j, i] = (0,0,0)
            else:
                colors[j, i] = palette[iter]

    plt.figure(figsize=size, dpi=dpi)
    plt.imshow(colors, zorder=1, interpolation='none')
    ax = plt.gca()
    ax.set_yticks([0, y_dim / 2, y_dim])
    ax.set_yticklabels([range_im[0], (range_im[0]+range_im[1])/ 2, range_im[1]])
    ax.set_xticks([0, x_dim / 2, x_dim])
    ax.set_xticklabels([range_real[0], (range_real[0]+range_real[1])/ 2, range_real[1]])

def est_area(n_list,s_list,reps=50,range_real=(-2,.5),range_im=(-1.1,1.1),sampling='pure'):

    total_area = (range_real[1]-range_real[0])*(range_im[1]-range_im[0])
    area = np.zeros(len(n_list),len(s_list),reps)
    for n in n_list:
        for s in s_list:
            for i in range(reps):
                if sampling == 'pure':
                    sample_real = np.random.uniform(range_real[0],range_real[1],s)
                    sample_im = np.random.uniform(range_im[0],range_im[1],s)
                else:
                    sample_real = np.random.uniform(range_real[0], range_real[1], s)
                    sample_im = np.random.uniform(range_im[0], range_im[1], s)
                mandel = 0
                for s_r, s_im in (sample_real, sample_im):
                    iter = iterate(complex(s_r,s_im), n)
                    if iter == 0:
                        mandel += 1
                area[n][s][i] = mandel/s*total_area

