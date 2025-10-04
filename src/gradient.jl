
"""
    gradient!(grad, θ, model)

Compute gradient of negative log posterior with respect to θ in-place.
"""
function gradient!(grad::Vector{Float64}, θ::Vector{Float64}, model::SegmentedModel)
    n = length(model.y)
    n_breakpoints = model.n_breakpoints
    n_beta = n_breakpoints + 2
    
    β = θ[1:n_beta]
    ψ = θ[n_beta+1:n_beta+n_breakpoints]
    log_σ = θ[end]
    σ = exp(log_σ)
    σ2 = σ^2
    
    ŷ = predict(model.x, θ, n_breakpoints)
    residuals = model.y .- ŷ
    sum_sq_residuals = sum(residuals.^2)
    
    # Gradient w.r.t. β (likelihood + prior)
    grad[1] = -sum(residuals) / σ2 + (β[1] - model.intercept_prior.μ) / model.intercept_prior.σ^2
    grad[2] = -sum(residuals .* model.x) / σ2 + β[2] / model.slope_prior.σ^2
    
    for i in 1:n_breakpoints
        grad[i+2] = -sum(residuals .* max.(0, model.x .- ψ[i])) / σ2 + β[i+2] / model.slope_prior.σ^2
    end
    
    # Gradient w.r.t. ψ (uniform prior contributes 0)
    for i in 1:n_breakpoints
        indicators = Float64.(model.x .> ψ[i])
        grad[n_beta + i] = β[i+2] / σ2 * sum(residuals .* indicators)
    end
    
    # Gradient w.r.t. log_σ (likelihood + prior)
    # From likelihood: n - sum_sq_residuals / σ²
    # From prior on σ with Jacobian: -1 + derivative of -logpdf w.r.t. log_σ
    # For Exponential(λ): ∂/∂log_σ [λσ - log_σ] = λσ - 1
    grad_likelihood = n - sum_sq_residuals / σ2
    
    # Prior gradient (for Exponential)
    if model.σ_prior isa Exponential
        λ = 1.0 / mean(model.σ_prior)
        grad_prior = λ * σ - 1.0
    else  # Truncated distribution
        # Use finite differences for complex distributions
        ε = 1e-8
        grad_prior = (nll(vcat(θ[1:end-1], log_σ + ε), model) - 
                     nll(vcat(θ[1:end-1], log_σ - ε), model)) / (2ε) - grad_likelihood
    end
    
    grad[end] = grad_likelihood + grad_prior
    
    return grad
end