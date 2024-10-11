using DifferentialEquations
using Plots

include("lorenz.jl")

u0 = [1.0; 0.0; 0.0]
tspan = (0.0, 100.0)
prob = ODEProblem(lorenz!, u0, tspan)
sol = solve(prob)

plot(sol, idxs = (1, 2, 3))
