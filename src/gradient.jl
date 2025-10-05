
"""
    gradient!(grad, θ, model)

Compute gradient of negative log posterior with respect to θ in-place.
"""
function gradient!(grad, θ, model::SegmentedModel)
    n = length(model.y)
    n_beta = model.n_breakpoints + 2

    β = θ[1:n_beta]
    θ_ψ = θ[n_beta+1:n_beta+model.n_breakpoints]
    ψ = transform_to_ordered(θ_ψ)  # Transform to ordered space
    log_σ = θ[end]
    σ = exp(log_σ)
    σ2 = σ^2

    x = model.x

    # Compute predictions and residuals
    ŷ = predict(model, θ)
    residuals = model.y .- ŷ

    sum_sq_residuals = sum(residuals.^2)

    # Get prior parameters
    mean_y = mean(model.y)
    sd_y = std(model.y)

    # Gradient w.r.t. β (likelihood + prior)
    # Likelihood: ∂NLL/∂β[1] = -1/σ² * Σ residuals * 1
    # Prior: beta[1] ~ normal(mean_y, sd_y*2), ∂/∂β[1] = (β[1] - mean_y)/(sd_y*2)^2
    prior_var_intercept = (sd_y * 2)^2
    grad[1] = -sum(residuals) / σ2 + (β[1] - mean_y) / prior_var_intercept

    # ∂NLL/∂β[2] = -1/σ² * Σ residuals * x
    # Prior: beta[2] ~ normal(0, 10), ∂/∂β[2] = β[2]/100
    grad[2] = -sum(residuals .* x) / σ2 + β[2] / 100.0

    # ∂NLL/∂β[i+2] = -1/σ² * Σ residuals * max(0, x - ψ[i])
    # Prior: beta[i+2] ~ normal(0, 10), ∂/∂β[i+2] = β[i+2]/100
    for i in 1:model.n_breakpoints
        grad[i+2] = -sum(residuals .* max.(0, x .- ψ[i])) / σ2 + β[i+2] / 100.0
    end

    # Gradient w.r.t. ψ (in ordered space)
    # ∂NLL/∂ψ[i] = β[i+2]/σ² * Σ residuals * 𝟙(x > ψ[i])
    grad_ψ = Vector{Float64}(undef, model.n_breakpoints)
    for i in 1:model.n_breakpoints
        indicators = Float64.(x .> ψ[i])
        grad_ψ[i] = β[i+2] / σ2 * sum(residuals .* indicators)
    end

    # Transform gradient to unconstrained space using chain rule
    # ψ[1] = θ_ψ[1], ψ[i] = ψ[i-1] + exp(θ_ψ[i])
    # ∂ψ[j]/∂θ_ψ[1] = 1 for all j
    # ∂ψ[j]/∂θ_ψ[i] = exp(θ_ψ[i]) for all j ≥ i (i > 1)
    for i in 1:model.n_breakpoints
        if i == 1
            # ∂L/∂θ_ψ[1] = Σ_j ∂L/∂ψ[j] * ∂ψ[j]/∂θ_ψ[1] = Σ_j ∂L/∂ψ[j]
            grad[n_beta + i] = sum(grad_ψ)
        else
            # ∂L/∂θ_ψ[i] = Σ_{j≥i} ∂L/∂ψ[j] * exp(θ_ψ[i])
            # Add Jacobian gradient: ∂/∂θ_ψ[i][log|det(J)|] = 1
            # Since we subtract log|det(J)| from NLL, we subtract 1 from gradient
            grad[n_beta + i] = sum(grad_ψ[i:end]) * exp(θ_ψ[i]) - 1.0
        end
    end

    # Gradient w.r.t. log_σ
    # Likelihood: ∂NLL/∂log_σ = n - sum_sq_residuals/σ²
    # Prior: sigma ~ exponential(10), ∂/∂log_σ[10*σ] = 10*σ
    grad[end] = n - sum_sq_residuals / σ2 + 10.0 * σ

    return grad
end
