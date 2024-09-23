using DifferentialEquations
using Plots


function mackey_glass_model!(du, u, h, p, t)
    a, b, tau = p
    x = u[1]
    delayedx = h(p, t - tau)[1]
    du[1] = -b*x + a*delayedx/(1 + delayedx^10)
end


history(p, t) = 0.5 + 0.02*t

a = 0.2
b = 0.1
tau = 17

p = (a, b, tau)
lags = [tau]
tspan = (0.0, 500.0)
u0 = [0.5]

prob = DDEProblem(mackey_glass_model!, u0, history, tspan, p; constant_lags = lags)
alg = MethodOfSteps(Tsit5())
sol = solve(prob, alg, abstol = 1e-11, reltol = 1e-8)
plot(sol, title = "Mackey-Glass", xaxis = "t", yaxis = "x", legend = false)
savefig("mackey_glass_solution.svg")
