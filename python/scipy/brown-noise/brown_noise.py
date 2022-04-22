#
# Generate brown noise.
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import tukey
from scipy.io import wavfile


T = 10.0    # signal duration, seconds
fs = 44100  # sample rate, Hz

nsamples = int(fs*T)

rng = np.random.default_rng(0x1ce1cebab1e)

# "Integrate" normal samples with cumsum().
x = rng.normal(size=nsamples).cumsum()

# Remove linear trend.
y = x - np.linspace(0, 1, nsamples)*(x[-1] - x[0])

# Remove mean.
y = y - y.mean()

# Normalize scale so the maximum absolute values is 1.0.
y = y / np.max(np.abs(y))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Generate a plot of the spectrum of y.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

f = rfft(y)
freq = rfftfreq(nsamples, d=1/fs)

absf = np.abs(f)

# dB full scale
dbfs = 20*np.log10(absf/np.max(absf))

plt.semilogx(freq, dbfs, alpha=0.8)
plt.xlabel('Frequency (Hz)')
plt.ylabel('dBfs')
plt.grid()
plt.xlim(2/T, 1.5*fs/2)
plt.ylim(-120, 10)
plt.title('Spectrum of a sample of brown noise')

# plt.show()
plt.savefig('brown-spectrum.svg')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Write y2 to a WAV file.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Create a window (i.e envelope) function to give the signal a smooth
# rise in volume the beginning and fall at the end.  This avoids the
# pop the can occur at the beginning and end of the sound when the WAV
# file is played.  (The window function doesn't have to be the tukey
# window; any function that provides an "attack-sustain-release" shape
# would work just as well.)
transition_time = 0.5  # seconds
w = tukey(nsamples, 2*transition_time/T)
wavfile.write('brown.wav', rate=fs, data=w*y)
