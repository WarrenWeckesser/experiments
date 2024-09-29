using DifferentialEquations
using Plots

# This example is from "DKLAG6: A Code Based on Continuously Imbedded Sixth Order
# Runge-Kutta Methods for the Solution of State Dependent Functional Differential
# Equations", by S. P. Corwin, D. Sarafyan, and S. Thompson, and they give their
# source of the equation as "Developing a Delay Differential Equation Solver" by
# C. A. H. Paul (J. Appl. Num. Math., Vol. 9, pp. 403-414, 1992). 

function exact_solution(t)
    # This function assumes 0 <= t <= 10
    e = Base.MathConstants.e
    if t <= e - 1
        t + 1
    elseif t <= e*e -1
        exp((t + 1)/e)
    else
        (e/(3 - log(t + 1)))^e
    end
end


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

plot(sol, title = "State-dependent delay example", xaxis = "t", yaxis = "y",
     color = "black", label = "Computed solution")

t = range(0.0, 10.0, 501)
y = [exact_solution(t1) for t1 = t]

plot!(t, y, label="Exact solution", linewidth = 5, alpha = 0.4)

savefig("paul1992.svg")
