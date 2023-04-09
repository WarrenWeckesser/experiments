
import numpy as np

from traits.api import (Float, List, Either, Array, Tuple, Bool,
                        Int, Any, on_trait_change)
from enable.api import Component


class LEDVUMeter(Component):

    values = Either(Float, Array)

    history_length = Int(64)
    history = List
    history_color = Any

    colors = List(Tuple)

    value_color = Any

    # The maximum value to be displayed in the VU Meter.
    # If log_scale is True (the default), this value must be greater than 0.
    max_value = Float(1.0)

    # The minimum value to be displayed in the VU Meter.
    # If log_scale is True (the default), this value must be greater than 0.
    min_value = Float(0.01)

    log_scale = Bool(True)

    bgcolor = (0, 0, 0)

    def _colors_default(self):
        colors = ([(0.0, 0.8, 0.0)] * 16 +
                  [(1.0, 1.0, 0.0)] * 2 +
                  [(1.0, 0.0, 0.0)] * 2)
        return colors

    def _history_color_default(self):
        return (0.5, 0.5, 0.5)

    def _value_color_default(self):
        return "highlight"

    #---------------------------------------------------------------------
    # Trait change handlers
    #---------------------------------------------------------------------

    @on_trait_change('values, colors, max_value, min_value, '
                     'log_scale, bgcolor')
    def _anytrait_changed(self, name, old, new):

        self.request_redraw()

    #---------------------------------------------------------------------
    # Component API
    #---------------------------------------------------------------------

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        clipped_values = np.asarray(self.values).clip(self.min_value,
                                    self.max_value)

        space = 3

        # Map clipped_values to an array of values in [0, 1].
        if self.log_scale:
            log_min = np.log10(self.min_value)
            log_max = np.log10(self.max_value)
            values = ((np.log10(clipped_values) - log_min) /
                      (log_max - log_min))
        else:
            values = ((clipped_values - self.min_value) /
                      (self.max_value - self.min_value))

        w = self.width
        h = self.height

        nleds = values.size
        nlevels = len(self.colors)

        led_width = (w - space * (nleds + 1)) / float(nleds)
        led_height = (h - space * (nlevels + 1)) / float(nlevels)

        levels = (nlevels * values).astype(int)

        with gc:
            gc.set_antialias(True)

            alpha = .25
            #gc.set_fill_color(self.history_color)
            for old_levels in reversed(self.history):
                gc.set_alpha(alpha)
                for j in range(nleds):
                    for k in range(levels[j], old_levels[j]):
                        x = space + j * (led_width + space)
                        y = space + k * (led_height + space)
                        color = self.colors[k]
                        gc.set_fill_color(color)
                        gc.rect(x, y, led_width, led_height)
                        gc.fill_path()
                alpha *= 0.9

            for j in range(nleds):
                for k in range(levels[j]):
                    x = space + j * (led_width + space)
                    y = space + k * (led_height + space)
                    color = self.colors[k]
                    gc.set_fill_color(color)
                    gc.rect(x, y, led_width, led_height)
                    if self.value_color == "highlight":
                        if k < levels[j] - 1:
                            gc.set_alpha(0.75)
                        else:  # k == levels[j]:
                            gc.set_alpha(1)
                    else:
                        gc.set_alpha(1)
                        if k == levels[j]:
                            gc.set_fill_color(self.value_color)
                    gc.fill_path()

        self.history.append(levels)
        if len(self.history) > self.history_length:
            self.history = self.history[1:]
