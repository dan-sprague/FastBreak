"""
    transform_to_ordered(θ_unconstrained)

Transform unconstrained breakpoint parameters to ordered breakpoints.
Uses STAN's ordered parameter transformation:
- ψ[1] = θ[1] (unconstrained)
- ψ[i] = ψ[i-1] + exp(θ[i]) for i > 1

See: https://mc-stan.org/docs/2_24/reference-manual/ordered-vector.html
"""
function transform_to_ordered(θ_unconstrained::Vector{Float64})
    n = length(θ_unconstrained)
    if n == 0
        return Float64[]
    end

    ψ = Vector{Float64}(undef, n)
    ψ[1] = θ_unconstrained[1]
    for i in 2:n
        ψ[i] = ψ[i-1] + exp(θ_unconstrained[i])
    end
    return ψ
end

"""
    transform_from_ordered(ψ)

Transform ordered breakpoints to unconstrained space.
Inverse of transform_to_ordered.

See: https://mc-stan.org/docs/2_24/reference-manual/ordered-vector.html
"""
function transform_from_ordered(ψ::Vector{Float64})
    n = length(ψ)
    if n == 0
        return Float64[]
    end

    θ = Vector{Float64}(undef, n)
    θ[1] = ψ[1]
    for i in 2:n
        θ[i] = log(ψ[i] - ψ[i-1])
    end
    return θ
end

function predict(model::SegmentedModel, θ::Vector{Float64})
    n_beta = model.n_breakpoints + 2
    n_breakpoints = model.n_breakpoints

    x = model.x
    β = θ[1:n_beta]
    θ_ψ = θ[n_beta+1:n_beta+n_breakpoints]
    ψ = transform_to_ordered(θ_ψ)
        
    ŷ = fill(β[1], length(x))
    ŷ .+= β[2] .* x
    for i in 1:n_breakpoints
        ŷ .+= β[i+2] .* max.(0, x .- ψ[i])
    end
    
    return ŷ
end
function negativeloglikelihood(θ, model::SegmentedModel)
    n = length(model.y)
    n_breakpoints = model.n_breakpoints
    n_beta = n_breakpoints + 2

    β = θ[1:n_beta]
    θ_ψ = θ[n_beta+1:n_beta+n_breakpoints]
    ψ = transform_to_ordered(θ_ψ) 
    log_σ = θ[end]
    σ = exp(log_σ)

    # Compute predictions
    y_pred = fill(β[1], n)
    y_pred .+= β[2] .* model.x
    for i in 1:n_breakpoints
        y_pred .+= β[i+2] .* max.(0, model.x .- ψ[i])
    end

    # Likelihood: y ~ normal(mu, sigma)
    residuals = model.y .- y_pred
    sum_sq_residuals = sum(residuals.^2)

    neg_log_likelihood = n * log_σ + sum_sq_residuals / (2 * σ^2)

    # Priors (matching Stan model)
    mean_y = mean(model.y)
    sd_y = std(model.y)
    xmin, xmax = model.ψ_prior_range

    neg_log_prior = 0.0

    # sigma ~ exponential(10)
    # p(sigma) = lambda * exp(-lambda * sigma), lambda = 10
    # -log p(sigma) = -log(lambda) + lambda * sigma = -log(10) + 10 * sigma
    neg_log_prior += 10.0 * σ - log(10.0)

    # beta[1] ~ normal(mean_y, sd_y * 2)
    # -log p(beta[1]) = 0.5 * log(2π) + log(sd) + 0.5 * ((beta[1] - mean_y) / sd)^2
    prior_sd_intercept = sd_y * 2
    neg_log_prior += 0.5 * ((β[1] - mean_y) / prior_sd_intercept)^2

    # beta[2:(K+2)] ~ normal(0, 10)
    for i in 2:n_beta
        neg_log_prior += 0.5 * (β[i] / 10.0)^2
    end

    # psi ~ uniform(min_x, max_x)
    # -log p(psi) = log(max_x - min_x) for each, but constant so skip
    # Check if in bounds (return Inf if not)
    for i in 1:n_breakpoints
        if ψ[i] < xmin || ψ[i] > xmax
            return Inf
        end
    end

    # Stan https://mc-stan.org/docs/2_24/reference-manual/ordered-vector.html
    log_jacobian_det = n_breakpoints > 1 ? sum(θ_ψ[2:end]) : 0.0

    # MAP objective = negative log posterior (with Jacobian adjustment)
    return neg_log_likelihood + neg_log_prior - log_jacobian_det
end