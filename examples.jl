"""
FastBreak Examples

This file contains examples demonstrating the FastBreak package for Bayesian
segmented regression. It includes:

1. Population growth example with logistic dynamics
2. Sine wave example comparing MAP vs MCMC estimation
3. Performance comparison with Stan
"""

using Plots
using Distributions
using StatsBase
using LogExpFunctions
using Random
using FastBreak

#==============================================================================#
# Utility Functions
#==============================================================================#

"""
    plot_mcmc_results(model::SegmentedModel, chain; n_plot_points=200, legend=:best)

Create a plot of MCMC results with credible intervals and breakpoint estimates.

# Arguments
- `model::SegmentedModel`: the segmented regression model
- `chain::Chains`: MCMC chain from sample_mcmc
- `n_plot_points::Int`: number of points for plotting predictions (default: 200)
- `legend`: legend position (default: :best)

# Returns
- A Plots.jl plot object with data, posterior mean fit, 95% credible interval, and breakpoints
"""
function plot_mcmc_results(model::SegmentedModel, chain; n_plot_points=200, legend=:best)
    # Extract samples
    n_breakpoints = model.n_breakpoints
    n_beta = n_breakpoints + 2

    # Get parameter names
    beta_names = filter(x -> occursin("beta", string(x)), names(chain))
    psi_names = filter(x -> occursin("psi", string(x)), names(chain))

    # Extract arrays
    β_samples = Array(chain[:, beta_names, :])
    ψ_samples = Array(chain[:, psi_names, :])

    # Compute means and standard errors
    β_mean = vec(mean(β_samples, dims=1))
    ψ_mean = vec(mean(ψ_samples, dims=1))
    ψ_se = vec(std(ψ_samples, dims=1))

    # Prediction function
    function predict_segmented(x_pred, β, ψ)
        y = fill(β[1], length(x_pred))
        y .+= β[2] .* x_pred
        for i in 1:length(ψ)
            y .+= β[i+2] .* max.(0, x_pred .- ψ[i])
        end
        return y
    end

    # Generate predictions for each posterior sample
    x_plot = range(minimum(model.x), maximum(model.x), length=n_plot_points)
    n_samples = size(β_samples, 1)
    predictions = zeros(length(x_plot), n_samples)

    for i in 1:n_samples
        predictions[:, i] = predict_segmented(x_plot, β_samples[i, :], ψ_samples[i, :])
    end

    # Compute credible intervals
    lower = [quantile(predictions[i, :], 0.025) for i in 1:length(x_plot)]
    upper = [quantile(predictions[i, :], 0.975) for i in 1:length(x_plot)]
    y_mean = vec(mean(predictions, dims=2))

    # Compute breakpoint y-values
    breakpoint_y_samples = zeros(length(ψ_mean), n_samples)
    for i in 1:n_samples
        for j in 1:length(ψ_mean)
            x_bp = ψ_samples[i, j]
            breakpoint_y_samples[j, i] = predict_segmented([x_bp], β_samples[i, :], ψ_samples[i, :])[1]
        end
    end
    breakpoint_y = vec(mean(breakpoint_y_samples, dims=2))

    # Create the plot
    p = scatter(model.x, model.y, label="Data", alpha=0.3, legend=legend)

    plot!(p, x_plot, y_mean,
          ribbon=(y_mean .- lower, upper .- y_mean),
          fillalpha=0.3, label="MCMC Posterior (95% CI)", lw=2,
          color = :crimson,linewidth = 1.5)

    scatter!(p, ψ_mean, breakpoint_y,
             label="Posterior Breakpoints", ms=5, alpha=1.0,markershape=:circle,
             linewidth=1.5,
             markercolor=:black,
             xerror=1.96 .* ψ_se)

    xlabel!(p, "x")
    ylabel!(p, "y")

    return p
end

"""
    noisy_sin(n=100; kwargs...)

Generate noisy sinusoidal data for testing.

# Arguments
- `n::Int`: number of data points
- `amplitude::Float64`: sine wave amplitude (default: 1.0)
- `frequency::Float64`: sine wave frequency (default: 1.0)
- `phase::Float64`: phase shift (default: 0.0)
- `noise_level::Float64`: standard deviation of Gaussian noise (default: 2.0)
- `xmin::Float64`: minimum x value (default: 0.0)
- `xmax::Float64`: maximum x value (default: 2π)
- `seed::Union{Int, Nothing}`: random seed for reproducibility (default: nothing)

# Returns
- `x::Vector{Float64}`: x coordinates
- `y::Vector{Float64}`: noisy y values
"""
function noisy_sin(n=100;
                   amplitude=1.0,
                   frequency=1.0,
                   phase=0.0,
                   noise_level=2.0,
                   xmin=0.0,
                   xmax=2π,
                   seed=nothing)

    seed !== nothing && Random.seed!(seed)

    x = range(xmin, xmax, length=n)
    y = amplitude .* sin.(frequency .* x .+ phase) .+ noise_level .* randn(n)

    return collect(x), y
end

"""
    Population

Logistic population growth model parameters.

# Fields
- `r::Float64`: intrinsic growth rate
- `P::Int64`: initial population
- `K::Int64`: carrying capacity
"""
struct Population
    r::Float64
    P::Int64
    K::Int64
end

"""
    du(u, p::Population)

Compute logistic growth rate.
"""
du(u, p::Population) = p.r * u * (1 - u / p.K)

"""
    simulate_population!(p::Population, u0::Int, tspan::Tuple{Int, Int}, dt::Float64)

Simulate stochastic logistic population growth using Poisson process.

# Arguments
- `p::Population`: population parameters
- `u0::Int`: initial population size
- `tspan::Tuple{Int, Int}`: time span (start, end)
- `dt::Float64`: time step

# Returns
- `t::StepRangeLen`: time points
- `u::Vector{Int}`: population size at each time point
"""
function simulate_population!(p::Population, u0::Int, tspan::Tuple{Int, Int}, dt::Float64;
                              env_noise::Float64=1.5,
                              demographic_noise_scale::Float64=1.0)
    t = tspan[1]:dt:tspan[2]
    u = zeros(Int, length(t))
    u[1] = u0

    for i in 2:length(t)
        # Environmental stochasticity (affects all individuals)
        r_noisy = p.r * exp(env_noise * randn())
        
        # Separate birth and death with scaled demographic noise
        birth_rate = r_noisy * u[i-1] * dt * demographic_noise_scale
        death_rate = (r_noisy * u[i-1]^2 / p.K) * dt * demographic_noise_scale
        
        births = rand(Poisson(max(0.0, birth_rate)))
        deaths = rand(Poisson(max(0.0, death_rate)))
        
        u[i] = max(0, u[i-1] + births - deaths)
    end
    return t, u
end

#==============================================================================#
# Example 1: Population Growth with Logistic Dynamics
#==============================================================================#

println("\n" * "="^80)
println("Example 1: Population Growth")
println("="^80)

# Simulate two different population trajectories
p = Population(0.04, 20, 100)  # r=0.1, P=10, K=100
tspan = (0, 100)
dt = 1.0

# Trajectory 1: Starting from u0=6
println("\nSimulating trajectory 1 (u0=6)...")
t1, u1 = simulate_population!(p, 10, tspan, dt)
model1 = SegmentedModel(collect(t1), u1, 1)

println("Running MCMC (2000 samples, 1000 warmup)...")
@time chain1 = sample_mcmc(model1, n_samples=2000, n_adapts=1000)

p1 = plot_mcmc_results(model1, chain1; legend=false)
title!(p1, "Population Growth",titleposition=:left)

# Trajectory 2: Starting from u0=2
println("\nSimulating trajectory 2 (u0=2)...")
t2, u2 = simulate_population!(p, 2, tspan, dt)
model2 = SegmentedModel(collect(t2), u2, 2)

println("Running MCMC (2000 samples, 1000 warmup)...")
@time chain2 = sample_mcmc(model2, n_samples=2000, n_adapts=1000)

p2 = plot_mcmc_results(model2, chain2; legend=false)
title!(p2, "")

# Combine plots
println("\nSaving population growth plots...")
p2 = plot(p1, p2, layout=@layout([a  b]), size=(800, 300), dpi=600,bottommargin=5Plots.mm)
savefig("img/population_growth_mcmc_final_morenoise3.svg")
println("Saved to img/population_growth_mcmc.svg")

#==============================================================================#
# Example 2: Sine Wave - MAP vs MCMC Comparison
#==============================================================================#

println("\n" * "="^80)
println("Example 2: Sine Wave - MAP vs MCMC Comparison")
println("="^80)

# Generate noisy sine data
println("\nGenerating noisy sine wave data...")
x, y = noisy_sin(200, amplitude=5.0, frequency=2.0, noise_level=5.0, seed=42)
model_sine = SegmentedModel(x, y, 4)

# Fit using MAP estimation
println("\nFitting MAP estimate...")
@time results_map = FastBreak.fit!(model_sine, show_trace=false, max_iter=1000)

# Fit using MCMC
println("\nRunning MCMC (2000 samples, 1000 warmup)...")
@time chain_sine = sample_mcmc(model_sine, n_samples=2000, n_adapts=1000,δ = 0.99)

# Create comparison plot
println("\nCreating comparison plot...")
x_plot = range(minimum(x), maximum(x), length=200)

# Start with MCMC results
p_sine = plot_mcmc_results(model_sine, chain_sine; legend=:topright,)
title!(p_sine, "Sine Wave: MLE/MAP vs MCMC")

# Add MAP fit
plot!(p_sine, x_plot, results_map(x_plot),
      label="MAP Estimate", lw=2, ls=:dash, color=:blue, alpha=1.0,
      legend = :bottomleft)

scatter!(p_sine, results_map.θ.ψ, results_map(collect(results_map.θ.ψ)),
         label="MAP Breakpoints", ms=4, mc=:blue, alpha=1.0,
         xerror=1.96 .* results_map.ψ_se)
plot(p_sine,size = (600,400),titlefontsize=12,legendfontsize=8,titlelocation=:left)
println("Saving sine wave comparison plot...")
savefig(p_sine, "img/sine_map_vs_mcmc_final_fixed.svg")
println("Saved to img/sine_map_vs_mcmc.png")





# Generate noisy sine data
println("\nGenerating noisy sine wave data...")
x, y = noisy_sin(500, amplitude=5.0, xmax=50,frequency=0.5, noise_level=1.5, seed=42)
model_sine = SegmentedModel(x, y,8)

# Fit using MAP estimation
println("\nFitting MAP estimate...")
@time results_map = FastBreak.fit!(model_sine, show_trace=false, max_iter=2000)

# Fit using MCMC
println("\nRunning MCMC (2000 samples, 1000 warmup)...")
@time chain_sine = sample_mcmc(model_sine, n_samples=2000, n_adapts=1000,δ = 0.99)

# Create comparison plot

println("\nCreating comparison plot...")
x_plot = range(minimum(x), maximum(x), length=200)

# Start with MCMC results
p_sine = plot_mcmc_results(model_sine, chain_sine; legend=:topright,)


title!(p_sine, "K = 8 breakpoints",titleposition=:left)

# Add MAP fit
plot!(p_sine, x_plot, results_map(x_plot),
      label="MAP Estimate", lw=2, ls=:dash, color=:blue, alpha=1.0,
      legend = :outertopright)


      scatter!(p_sine, results_map.θ.ψ, results_map(collect(results_map.θ.ψ)),
         label="MAP Breakpoints", ms=4, mc=:blue, alpha=1.0,
         xerror=1.96 .* results_map.ψ_se)
p_sine = plot(p_sine,size = (900,400),titlefontsize=12,legendfontsize=8,titlelocation=:left)
println("Saving sine wave comparison plot...")
savefig(p_sine, "img/sine_map_vs_mcmc_final_low_complicated2.svg")
