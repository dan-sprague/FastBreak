module FastBreak

using Optim
using LinearAlgebra
using Printf
using StatsBase: mean, quantile, std
using AdvancedHMC
using MCMCChains
using Distributions 
using JSON

#include("dist.jl")



"""
Segmented regression model with data and priors
"""
struct SegmentedModel
    x::Vector{Float64}
    y::Vector{Float64}
    n_breakpoints::Int
    ψ_prior_range::Tuple{Float64, Float64}   # Uniform prior bounds for breakpoints
end

"""
    SegmentedModel(x, y, n_breakpoints; kwargs...)

Create a segmented regression model with data and priors.

# Arguments
- `x::Vector`: predictor variable
- `y::Vector`: response variable  
- `n_breakpoints::Int`: number of breakpoints
- `slope_prior::Normal`: prior for slope parameters (default: Normal(0, 1))
- `intercept_prior::Normal`: prior for intercept (default: Normal(0, 10))
- `σ_prior::Distribution`: prior for noise std (default: Exponential(1))
- `ψ_prior_range::Tuple`: uniform prior bounds for breakpoints (default: (min(x), max(x)))
"""
function SegmentedModel(
    x::Vector{<:Real}, 
    y::Vector{<:Real},
    n_breakpoints::Int;
    ψ_prior_range::Union{Nothing, Tuple{<:Real, <:Real}}=nothing
)
    # Default ψ_prior_range to data range
    if isnothing(ψ_prior_range)
        ψ_prior_range = (minimum(x), maximum(x))
    end
    
    return SegmentedModel(
        Float64.(x),
        Float64.(y),
        n_breakpoints,
        Float64.(ψ_prior_range)
    )
end

""" 
Fitted model parameters
"""
struct FittedParams
    ψ::Vector{Int}
    β::Vector{Float64}
    σ::Float64
end


struct FittedSegmentModel
    θ::FittedParams
    β_se::Vector{Float64}
    β_ci::Matrix{Float64}
    ψ_se::Vector{Float64}
    ψ_ci::Matrix{Float64}
    σ_se::Float64
    σ_ci::Tuple{Float64, Float64}
    covariance_matrix::Matrix{Float64}
    correlation_matrix::Matrix{Float64}
    hessian::Matrix{Float64}
    optim_result::Optim.MultivariateOptimizationResults
end

ψ(x::FittedParams) = x.ψ
β(x::FittedParams) = x.β
σ(x::FittedParams) = x.σ

function (m::FittedSegmentModel)(x)
    y = fill(β(m.θ)[1], length(x))  # Intercept
    y .+= β(m.θ)[2] .* x             # First slope
    
    for i in 1:length(ψ(m.θ))
        y .+= β(m.θ)[i+2] .* max.(0, x .- ψ(m.θ)[i])
    end
    return y
end

include("univariate.jl")
include("gradient.jl")
include("hessian.jl")
include("output.jl")
include("stan_utils.jl")



export SegmentedModel, FittedSegmentModel, FittedParams, fit!, predict, print_results, write_stan_data

end # module FastBreak


