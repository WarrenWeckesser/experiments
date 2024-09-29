using DifferentialEquations
using Plots

# This example is from "DKLAG6: A Code Based on Continuously Imbedded Sixth Order
# Runge-Kutta Methods for the Solution of State Dependent Functional Differential
# Equations", by S. P. Corwin, D. Sarafyan, and S. Thompson, and they give their
# source of the equation as "Developing a Delay Differential Equation Solver" by
# C. A. H. Paul (J. Appl. Num. Math., Vol. 9, pp. 403-414, 1992). 

function paul1992!(du, u, h, p, t)
    y = u[1]
    delayedy = h(p, log(y) - 1)[1]
    du[1] = (1/(t + 1))*y*delayedy
end


history(p, t) = [1.0]
lag(u, p, t) = t - (log(u[1]) - 1.0)

tspan = (0.0, 10.0)
u0 = [1.0]

prob = DDEProblem(paul1992!, u0, history, tspan; dependent_lags = [lag])
alg = MethodOfSteps(Tsit5())
sol = solve(prob, alg, abstol = 1e-11, reltol = 1e-8)

plot(sol, title = "State-dependent delay example", xaxis = "t", yaxis = "x", legend = false)
savefig("state_dependent_delay_solution.svg")
