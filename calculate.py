import pyfftw
import numpy as np

a = pyfftw.interfaces.numpy_fft.fftfreq(4096, d=1./44100)
n = np.arange(4096)
i = np.hstack((n[:,np.newaxis], a[:,np.newaxis]))[:2047]

print(i)