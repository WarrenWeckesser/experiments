using DifferentialEquations
using Plots
import PlotlyJS

plotlyjs()

#
# The exact solution in the interval [0, tau] when the initial
# conditions are [S(0), I(0), R(0)] = [1-I0, I0, 0]
#
function history(p, t)
    alpha, tau, I0 = p
    I = I0 ./ (I0 .+ (1-I0)*exp.(-alpha*t))
    return hcat(1 .- I, I, zeros(size(t)))
end


function SIR_delay_model!(du, u, h, p, t)
    alpha, tau, I0 = p
    S, I, R = u
    delayedS, delayedI, delayedR = h(p, t - tau)
    du[1] = -alpha*S*I
    du[2] = alpha*(S*I - delayedS*delayedI)
    du[3] = alpha*delayedS*delayedI
end


alpha = 0.5
tau = 4.0
I0 = 0.001

p = (alpha, tau, I0)
lags = [tau]

# At t=0, an initial population of size I0 got infected. This initial
# cohort will recover at t=tau, and move to the Recovered state.
# In the interval [0, tau], the exact solution is implemented in
# history(p, t).  At the end of this interval, I jumps down by
# I0 and R jumps up by I0, reflecting the recovery of the initial cohort.
# We will solve the DDEs on the interval [tau, 50], with the known solution
# on [0, tau] becoming the initial history for the DDE solver.

tspan = (tau, 50.0)
u0 = history(p, tau) + [0 (-I0) I0]

prob = DDEProblem(SIR_delay_model!, u0, history, tspan, p; constant_lags = lags)
alg = MethodOfSteps(Tsit5())
sol = solve(prob, alg, abstol = 1e-11, reltol = 1e-8)

plot(sol, title = "SIR delay model", xaxis = "t", label = ["S" "I" "R"],
     ticks = :native, line_color = [:green, :red, :blue])
xlims!(0, tspan[2])
t_initial_interval = range(0, tau, 50)
u_initial_interval = history(p, t_initial_interval)
# I don't know how to make the colors of these lines match those of the previous
# plot() call.  `line_color` seemed to be ignored in this call.
plot!(t_initial_interval, u_initial_interval, label = ["S initial" "I initial" "R initial"])
