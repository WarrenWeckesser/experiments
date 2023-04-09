
import random

from traits.api import HasTraits, Instance, Any
from traitsui.api import View, UItem, HGroup
from enable.api import ComponentEditor
try:
    from pyface.timer.api import Timer
except ImportError:
    import sys
    print >> sys.stderr, "Unable to import the pyface package.  This demo requires pyface."
    sys.exit(-1)

from enable.gadgets.vu_meter import VUMeter


class Demo(HasTraits):

    vu1 = Instance(VUMeter)
    vu2 = Instance(VUMeter)

    timer = Any

    traits_view = \
        View(
            HGroup(
                UItem('vu1', editor=ComponentEditor(size=(60, 60)),
                             style='custom'),
                UItem('vu2', editor=ComponentEditor(size=(60, 60)),
                             style='custom'),
            ),
            width=750,
            height=225,
            title="VU Meter",
            resizable=True,
        )

    def __init__(self, *args, **kwds):
        super(self.__class__, self).__init__(*args, **kwds)
        self.animate()

    def animate(self):
        if self.timer is None:
            self.timer = Timer(100, self.random_step)
        else:
            self.timer.Start()

    def random_step(self):
        delta1 = (5 * (2 * random.random() - 1) + 0.02 * (65 - vu1.percent) +
                  0.01 * (vu2.percent - vu1.percent))
        vu1.percent = min(125, max(0, vu1.percent + delta1))
        delta2 = 3 * (2 * random.random() - 1) + 0.005 * (80 - vu2.percent)
        vu2.percent = min(125, max(0, vu2.percent + delta2))


if __name__ == "__main__":
    color = (0.95, 0.93, 0.85)
    vu1 = VUMeter(border_visible=True, border_width=2, bgcolor=color,
                  text="LEFT")
    vu2 = VUMeter(border_visible=True, border_width=2, bgcolor=color,
                  text="RIGHT")

    demo = Demo(vu1=vu1, vu2=vu2)
    ##demo.edit_traits()
    demo.configure_traits()
