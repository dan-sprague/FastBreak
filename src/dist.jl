struct Normal
    μ::Float64
    σ::Float64
end

struct Exponential
    θ::Float64  # Mean parameter (to match Distributions.jl convention)
end

# Helper functions to replace Distributions.jl functionality
function logpdf(d::Normal, x::Float64)
    return -0.5 * log(2π) - log(d.σ) - 0.5 * ((x - d.μ) / d.σ)^2
end

function logpdf(d::Exponential, x::Float64)
    λ = 1.0 / d.θ  # Rate parameter
    return log(λ) - λ * x
end

function mean(d::Exponential)
    return d.θ
end

function rand(d::Normal, n::Int)
    return d.μ .+ d.σ .* randn(n)
end

function quantile(d::Normal, p::Float64)
    # For standard normal N(0,1), use inverse error function
    # For N(μ, σ), transform: μ + σ * Φ⁻¹(p)
    z = sqrt(2) * erfinv(2*p - 1)
    return d.μ + d.σ * z
end