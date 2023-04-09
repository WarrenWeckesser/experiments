
# XXX This demo is currently broken.  Too much hacking in led_vu_meter.py
# while working on led_vu_spectrum.py left this code unusable.

import numpy as np

from traits.api import HasTraits, Instance, Array, Float
from traitsui.api import View, UItem, Item, RangeEditor, Group, VGroup, HGroup
from enable.api import ComponentEditor

from led_vu_meter import LEDVUMeter

num_leds = 80

class Demo(HasTraits):

    vu = Instance(LEDVUMeter)

    #x = Array

    a = Float(1.0)
    w = Float(1.0)

    traits_view = \
        View(
            HGroup(
                VGroup(
                    Group(
                        UItem('vu', editor=ComponentEditor(size=(20, 20)),
                                     style='custom', width=600, height=300),
                    ),
                Item('a', editor=RangeEditor(low=0.0, high=10.0,
                                             mode='slider')),
                Item('w', editor=RangeEditor(low=0.0, high=10.0,
                                             mode='slider')),
                ),
            ),
            width=650,
            height=200,
            title="LED VU Meter",
            resizable=True,
        )

    def _anytrait_changed(self):
        t = np.linspace(0, np.pi, num_leds)
        y = 0.5 * self.a * (1.0 + np.sin(self.w * t))
        vu.values = y


if __name__ == "__main__":
    vu = LEDVUMeter(values=np.ones(num_leds), max_value=10, log_scale=False)
    vu = LEDVUMeter(values=np.ones(num_leds),
                    min_value=0, #max_value=32.0,
                    #colors=colors,
                    history_length=0,
                    value_color=(0, 1.0, 0.1),
                    #history_color=(0.3, 0.5, 0.4),
                    log_scale=False,
                    )
    print vu
    demo = Demo(vu=vu)
    ##demo.edit_traits()
    print demo
    demo.configure_traits()
