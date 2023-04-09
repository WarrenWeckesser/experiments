# LED-like spectrogram.
# Requires pyaudio.  If pyaudio is not installed, fake data is generated for
# the display.

from __future__ import print_function

import sys
import time

import numpy as np
from numpy.fft import fft
from matplotlib.mlab import psd

fake_data = False
try:
    import pyaudio
except ImportError:
    fake_data = True
    print('pyaudio is not available, so random data will be used.')

from traits.api import HasTraits, Instance, Any, Float
from traitsui.api import View, UItem
from enable.api import ComponentEditor

try:
    from pyface.timer.api import Timer
except ImportError:
    print("Unable to import the pyface package.  This demo requires pyface.",
          file=sys.stderr)
    sys.exit(-1)

# Local import
from led_vu_meter import LEDVUMeter


NUM_SAMPLES = 1024
#SAMPLING_RATE = 11025
#SAMPLING_RATE = 22050
SAMPLING_RATE = 44100
#SAMPLING_RATE = 48000

_stream = None

if fake_data:
    def get_audio_data(num_reps=2):
        time.sleep(0.1)
        x1 = (0.0005*np.abs(np.random.randn(16)).cumsum()**3)[::-1]
        x2 = (0.002*np.abs(np.random.randn(16).cumsum())**3)[::-1]
        return x1 + x2
else:
    def get_audio_data(num_reps=2):
        global _stream
        if _stream is None:
            pa = pyaudio.PyAudio()
            _stream = pa.open(format=pyaudio.paInt16, channels=1,
                              rate=SAMPLING_RATE, input=True,
                              frames_per_buffer=NUM_SAMPLES)

        ds = 16
        data = np.zeros(ds)
        #data = np.zeros(NUM_SAMPLES / 2 / ds)
        for k in range(num_reps):
            try:
                audio_data = np.fromstring(_stream.read(NUM_SAMPLES),
                                            dtype=np.short)
            except IOError:
                audio_data = np.ones((NUM_SAMPLES,), dtype=np.short)
            normalized_data = audio_data / 32768.0
            spec = fft(normalized_data)[:NUM_SAMPLES / 2]
            z = (spec * spec.conj()).real
            binedges = np.round(np.logspace(np.log10(2), np.log10(NUM_SAMPLES/2+1), ds+1)).astype(int) - 1
            print("binegdges:", binedges)
            print("z.shape:", z.shape)
            binsums = np.add.reduceat(z, binedges[:-1])
            m = np.sqrt(binsums / np.diff(binedges))
            #data += np.sqrt(z.mean(axis=-1))
            data += m
        return data / num_reps


"""
def get_audio_dataXX(num_reps=1):
    global _stream
    if _stream is None:
        pa = pyaudio.PyAudio()
        _stream = pa.open(format=pyaudio.paInt16, channels=1,
                          rate=SAMPLING_RATE, input=True,
                          frames_per_buffer=NUM_SAMPLES)

    try:
        audio_data = np.fromstring(_stream.read(NUM_SAMPLES), dtype=np.short)
    except IOError:
        audio_data = np.ones((NUM_SAMPLES,), dtype=np.short)
    normalized_data = audio_data / 32768.0
    data, freqs = psd(normalized_data, NFFT=72)
    print data.min(), data.max()
    return data / num_reps
"""


class Demo(HasTraits):

    vu = Instance(LEDVUMeter)

    timer = Any

    traits_view = \
        View(
            UItem('vu', editor=ComponentEditor(size=(20, 20)),
                        style='custom'),
            width=800,
            height=225,
            title="LED VU Meter",
            resizable=True,
        )

    def __init__(self, *args, **kwds):
        super(self.__class__, self).__init__(*args, **kwds)
        self.animate()

    def animate(self):
        if self.timer is None:
            self.timer = Timer(25, self.read_mic)
        else:
            self.timer.Start()

    def read_mic(self):
        data = get_audio_data(num_reps=3)
        self.vu.values = data


if __name__ == "__main__":
    colors = ([(0.1, 0.7, 0.2)] * 24 +
              [(1.0, 1.0, 0.1)] * 4 +
              [(1.0, 0.1, 0.1)] * 4)

    vu = LEDVUMeter(values=get_audio_data(),
                    min_value=1e-2, max_value=32.0,
                    #colors=colors,
                    #value_color=(0, 1.0, 0.1),
                    history_color=(0.3, 0.5, 0.4),
                    )
    demo = Demo(vu=vu)
    ##demo.edit_traits()
    demo.configure_traits()
