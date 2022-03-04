
# Written for
# https://stackoverflow.com/questions/71342740/
# dry-wet-i-e-include-partial-raw-signal-parameter-on-iir-notch-filter-before-c

"""
First, a couple observations:

* To incorporate the "wet" factor, we can use a little algrebra.
  An IIR filter can be represented as the rational function

              P_B(z)
      H(z) =  ------
              P_A(z)

  where P_A(z) and P_B(z) the polynomials in z^-1.  A notch filter
  will have a gain of 0 at the center frequency, but you want to
  mix the "wet" and "dry" (i.e. filtered and unfiltered) signals.
  Let w be the "wet" fraction, 0 <= w <= 1, so w = 0 implies no
  filtering, w=0.25 implies the gain is 0.75 at the notch frequency,
  etc.  Such a filter can be expressed as

                              P_B(z)   (1 - w)*P_A(z) + w*P_B(z)
      H(z) =  (1 - w) + (w) * ------ = -------------------------
                              P_A(z)            P_A(z)


  In terms of the arrays of coefficients returned in a call such
  as `b, a = iirnotch(w0, Q, fs)`, the coefficients of the modified
  filter are `b_mod = (1-w)*a + w*b` and `a_mod = a`.

* `iirnotch(w0, Q, fs)` returns two 1-d array of length 3.  These
  are the coefficients of the "biquad" notch filter.  To create
  a new filter that is a cascade of several notch filters, you
  can simply stack the arrays returned by `iirnotch` into an
  array with shape (n, 6); this is the *SOS* (second order sections)
  format for a filter.  This format is actually the recommended
  format for filters beyond a few orders, because it is numerically
  more robust than the (b, a) format (which requires high order
  polynomials).
"""

import numpy as np
from scipy.signal import iirnotch, sosfreqz
import matplotlib.pyplot as plt


samplerate = 4000
notch_freqs = (220, 440, 880)
Q = 12.5

filters = [iirnotch(freq, Q, samplerate) for freq in notch_freqs]

# Stack the filter coefficients into an array with shape
# (len(notch_freqs, 6)).  This array of "second order sections"
# can be used with sosfilt, sosfreqz, etc.

# This version, `sos`, is the combined filter without taking
# into account the desired "wet" factor.  This is created just
# for comparison to the filter with "wet" factors.
sos = np.block([[b, a] for b, a in filters])

# sos2 includes the desired "wet" factor of the notch filters.
wet = (0.25, 0.5, 1.0)
sos2 = np.block([[(1-w)*a + w*b, a] for w, (b, a) in zip(wet, filters)])

# Compute the frequency responses of the two filters.
w, h = sosfreqz(sos, worN=8000, fs=samplerate)
w2, h2 = sosfreqz(sos2, worN=8000, fs=samplerate)

plt.subplot(2, 1, 1)
plt.plot(w, np.abs(h), '--', alpha=0.75, label='Simple cascade')
plt.plot(w2, np.abs(h2), alpha=0.8, label='With "wet" factor')
plt.title('Frequency response of cascaded notch filters')
for yw in wet:
    plt.axhline(1 - yw, color='k', alpha=0.4, linestyle=':')
plt.grid(alpha=0.25)
plt.ylabel('|H(f)|')
plt.legend(framealpha=1, shadow=True, loc=(0.6, 0.15))

plt.subplot(2, 1, 2)
plt.plot(w, np.angle(h), '--', alpha=0.75, label='Simple cascade')
plt.plot(w2, np.angle(h2), alpha=0.8, label='With "wet" factor')
plt.xlabel('frequency')
plt.grid(alpha=0.25)
plt.ylabel('∠H(f) (radians)')
plt.legend(framealpha=1, shadow=True, loc=(0.6, 0.15))
# plt.show()
plt.savefig('cascaded_notch_filters.svg', transparent=True)
