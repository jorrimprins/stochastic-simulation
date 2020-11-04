from numba import jit
import numpy as np
import seaborn as sns
import matplotlib as plt
sns.color_palette("magma", as_cmap=True)

@jit
def iterate(c, n):

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
def set_mb(range_real, range_im, size, n):
    ## Creates set for given ranges, range_real, range_im and size are tuples.
    range_real = np.linspace(range_real[0],range_real[1],size[0])
    range_im = np.linspace(range_im[0],range_im[1],size[1])
    mb_set = np.zeros(size)

    for i in range(size[0]):
        for j in range(size[1]):
            mb_set[i, j] = iterate(complex(range_real[i], range_im[j]), n)
    return mb_set

@jit
def view_mb(range_real, range_im, size, n, dpi=100):
    plotsize = [i * dpi for i in size]
    mb_set = set_mb(range_real, range_im, plotsize, n)

    fig = plt.figure(figsize=size, dpi=dpi)
    #ticks = np.arange(0, img_width, 3 * dpi)
    #x_ticks = xmin + (xmax - xmin) * ticks / img_width
    #plt.xticks(ticks, x_ticks)
    #y_ticks = ymin + (ymax - ymin) * ticks / img_width
    #plt.yticks(ticks, y_ticks)
    #norm = colors.PowerNorm(0.1)
    fig.plot(mb_set.T)#, origin='lower', norm=norm)
    fig.show()


