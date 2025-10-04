
"""
    gradient!(grad, θ, model)

Compute gradient of negative log posterior with respect to θ in-place.
"""
function gradient!(grad, θ, model::SegmentedModel)
    n = length(model.y)
    n_beta = model.n_breakpoints + 2

    β = θ[1:n_beta]
    ψ = θ[n_beta+1:n_beta+model.n_breakpoints]
    log_σ = θ[end]
    σ = exp(log_σ)
    σ2 = σ^2

    x = model.x
    
    # Compute predictions and residuals
    ŷ = predict(model, θ)
    residuals = model.y .- ŷ

    sum_sq_residuals = sum(residuals.^2)
    
    # Gradient w.r.t. β
    # ∂NLL/∂β[1] = -1/σ² * Σ residuals * 1
    grad[1] = -sum(residuals) / σ2
    
    # ∂NLL/∂β[2] = -1/σ² * Σ residuals * x
    grad[2] = -sum(residuals .* x) / σ2
    
    # ∂NLL/∂β[i+2] = -1/σ² * Σ residuals * max(0, x - ψ[i])
    for i in 1:model.n_breakpoints
        grad[i+2] = -sum(residuals .* max.(0, x .- ψ[i])) / σ2
    end
    
    # Gradient w.r.t. ψ
    # ∂NLL/∂ψ[i] = β[i+2]/σ² * Σ residuals * 𝟙(x > ψ[i])
    for i in 1:model.n_breakpoints
        indicators = Float64.(x .> ψ[i])
        grad[n_beta + i] = β[i+2] / σ2 * sum(residuals .* indicators)
    end
    
    # Gradient w.r.t. log_σ
    # ∂NLL/∂log_σ = n - sum_sq_residuals/σ²
    grad[end] = n - sum_sq_residuals / σ2
    
    return grad
end