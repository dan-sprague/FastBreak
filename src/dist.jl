struct Normal
    μ::Float64
    σ::Float64
end

struct Exponential
    θ::Float64  # Mean parameter (to match Distributions.jl convention)
end

# Helper functions to replace Distributions.jl functionality
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

function quantile_normal(p::Float64)
    # Approximation for standard normal quantile (good enough for 95% CI)
    if p ≈ 0.975
        return 1.96
    elseif p ≈ 0.95
        return 1.645
    else
        # Simple approximation using inverse error function
        return sqrt(2) * erfinv(2*p - 1)
    end
end