using DifferentialEquations
using Plots

# The example was taken from "High Order Methods for State-Dependent Delay
# Differential Equations with Nonsmooth Solutions", by A. Feldstein and
# K. W. Neves (SIAM J. Numer. Anal., Vol. 21, No. 5., pp. 844-863, October,
# 1984).

function feldsteinneves1984!(du, u, h, p, t)
    y = u[1]
    delayedy = h(p, y - sqrt(2))[1]
    du[1] = delayedy/(2*sqrt(t + 1))
end


function exact_solution(t)
    # This function assumes 0 <= t <= 2
    if t < 1
        sqrt(t + 1)
    else
        tp1 = t + 1
        tp1/4 + 0.5 + (1 - sqrt(2)/2)*sqrt(tp1)
    end
end


history(p, t) = [1.0]
lag(u, p, t) = t - (u[1]) - sqrt(2)

tspan = (0.0, 2.0)
u0 = [1.0]

prob = DDEProblem(feldsteinneves1984!, u0, history, tspan; dependent_lags = [lag])
alg = MethodOfSteps(Tsit5())
sol = solve(prob, alg, abstol = 1e-11, reltol = 1e-8)

plot(sol, title = "State Dependent Delay Example\nFeldstein-Neves (1984)",
     xaxis = "t", yaxis = "y", color="black", label = "Computed solution")

t = range(0.0, 2.0, 201)
y = [exact_solution(t1) for t1 = t]

plot!(t, y, label="Exact solution", linewidth = 5, alpha = 0.4)

savefig("feldstein_neves_1984.svg")
