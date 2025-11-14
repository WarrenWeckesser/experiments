import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation
from numpngw import AnimatedPNGWriter


def G(u, v, f, k):
    return f * (1 - u) - u*v**2


def H(u, v, f, k):
    return -(f + k) * v + u*v**2


def grayscott1d(y, t, f, k, Du, Dv, dx):
    """
    Differential equations for the 1-D Gray-Scott equations.

    The ODEs are derived using the method of lines.
    """
    # The vectors u and v are interleaved in y.  We define
    # views of u and v by slicing y.
    u = y[::2]
    v = y[1::2]

    # dydt is the return value of this function.
    dydt = np.empty_like(y)

    # Just like u and v are views of the interleaved vectors
    # in y, dudt and dvdt are views of the interleaved output
    # vectors in dydt.
    dudt = dydt[::2]
    dvdt = dydt[1::2]

    # Compute du/dt and dv/dt.  The end points and the interior points
    # are handled separately.
    dudt[0]    = G(u[0],    v[0],    f, k) + Du * (-2.0*u[0] + 2.0*u[1]) / dx**2
    dudt[1:-1] = G(u[1:-1], v[1:-1], f, k) + Du * np.diff(u,2) / dx**2
    dudt[-1]   = G(u[-1],   v[-1],   f, k) + Du * (- 2.0*u[-1] + 2.0*u[-2]) / dx**2
    dvdt[0]    = H(u[0],    v[0],    f, k) + Dv * (-2.0*v[0] + 2.0*v[1]) / dx**2
    dvdt[1:-1] = H(u[1:-1], v[1:-1], f, k) + Dv * np.diff(v,2) / dx**2
    dvdt[-1]   = H(u[-1],   v[-1],   f, k) + Dv * (-2.0*v[-1] + 2.0*v[-2]) / dx**2

    return dydt


def ic(L, n):
    # Hacked on this until I got an interesting solution. There is nothing special
    # about the formulas used here.
    x = np.linspace(0, L, n)
    u0 = 1.0 - 0.5*np.cos(2*np.pi*x/L) + 0.5*np.cos(0.5*2*np.pi*x/L) + 0.5*np.cos(3*2*np.pi*x/L)
    u0 = (u0 - u0.min())/np.ptp(u0)
    v0 = 0.125 + 0.2*np.cos(0.5*2*np.pi*x/L) + 0.125*np.cos(2*np.pi*x/L)
    y0 = np.empty(2*n, dtype=np.float64)
    y0[::2] = u0
    y0[1::2] = v0
    return y0


L = 24.0
T = 5000.0

# Spatial grid size.
nx = 2001
# Number of time samples.
nt = 1250

y0 = ic(L, nx)
c = 0.001
t0 = np.linspace(0, T + 1/c, nt)
t = t0 + np.expm1(-c*t0)/c

# Equation parameters.
f = 0.024
k = 0.055
Du = 0.05
Dv = 0.01
dx = L/(nx - 1)

sol, info = odeint(grayscott1d, y0, t,
                   args=(f, k, Du, Dv, dx),
                   ml=2, mu=2,
                   rtol=1e-9, atol=1e-12,
                   full_output=True)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Generate an animated PNG from the solution.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def update_line(num, x, data, t, uline, vline, time_text):
    """
    Animation "call back" function for each frame.
    """
    uline.set_data(x, data[num, ::2])
    vline.set_data(x, data[num, 1::2])
    time_text.set_text(f"t = {t[num]:8.2f}")
    return uline, vline


print("Generating the animated PNG file.")

fig = plt.figure(figsize=(7.25, 2.0))

# Initial plot.  The animation will update elements of this plot for each frame.
ax = fig.gca()
ax.set_title(f"1-D Gray-Scott Equation (L = {L})")
ax.set_xlim(-0.005*L, 1.005*L)
ax.set_ylim(-0.05, 1.1)
ax.grid(True, alpha=0.5)
time_text = ax.text(0.45, 0.8, f't = {0.0:8.2f}', transform=ax.transAxes)

x = np.linspace(0, L, nx)

step = 3
data = sol[::step]
tdata = t[::step]

# Plot the initial condition. lineplot is reused in the animation.
uline, = ax.plot(x, data[0, ::2], '-', linewidth=2.5, label='u')
vline, = ax.plot(x, data[0, 1::2], '--', linewidth=2.5, label='v')
ax.legend(framealpha=1, shadow=True, loc='upper left')
plt.tight_layout()

ani = animation.FuncAnimation(fig, update_line, frames=len(tdata),
                              init_func=lambda : None,
                              fargs=(x, data, tdata, uline, vline, time_text))
writer = AnimatedPNGWriter(fps=12)
ani.save('gray-scott-1d.png', dpi=60, writer=writer)

plt.close(fig)
