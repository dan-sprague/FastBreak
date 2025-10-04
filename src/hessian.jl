
"""
    hessian!(hess, θ, model)

Compute Hessian of negative log posterior with respect to θ in-place.
"""
function hessian!(hess, θ, model::SegmentedModel)
    n = length(model.y)
    n_beta = model.n_breakpoints + 2
    n_params = length(θ)

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
    
    # Precompute derivatives of prediction w.r.t. parameters
    # dŷ/dβ[1] = 1
    # dŷ/dβ[2] = x
    # dŷ/dβ[i+2] = max(0, x - ψ[i])
    # dŷ/dψ[i] = -β[i+2] * 𝟙(x > ψ[i])
    
    dy_dβ = zeros(n, n_beta)
    dy_dβ[:, 1] .= 1.0
    dy_dβ[:, 2] .= x
    for i in 1:model.n_breakpoints
        dy_dβ[:, i+2] .= max.(0, x .- ψ[i])
    end

    dy_dψ = zeros(n, model.n_breakpoints)
    for i in 1:model.n_breakpoints
        indicators = Float64.(x .> ψ[i])
        dy_dψ[:, i] .= -β[i+2] .* indicators
    end
    
    fill!(hess, 0.0)
    
    # Block 1: ∂²NLL/∂β[j]∂β[k] = 1/σ² * Σ (∂ŷ/∂β[j]) * (∂ŷ/∂β[k])
    for j in 1:n_beta
        for k in 1:n_beta
            hess[j, k] = sum(dy_dβ[:, j] .* dy_dβ[:, k]) / σ2
        end
    end
    
    # Block 2: ∂²NLL/∂β[j]∂ψ[k]
    for j in 1:n_beta
        for k in 1:model.n_breakpoints
            # = 1/σ² * Σ (∂ŷ/∂β[j]) * (∂ŷ/∂ψ[k])
            hess[j, n_beta + k] = sum(dy_dβ[:, j] .* dy_dψ[:, k]) / σ2
            
            # Add second derivative contribution for β[k+2] and ψ[k]
            if j == k + 2
                # ∂²ŷ/∂β[k+2]∂ψ[k] = -𝟙(x > ψ[k])
                indicators = Float64.(x .> ψ[k])
                hess[j, n_beta + k] += sum(residuals .* (-indicators)) / σ2
            end
            
            hess[n_beta + k, j] = hess[j, n_beta + k]  # Symmetry
        end
    end
    
    # Block 3: ∂²NLL/∂ψ[j]∂ψ[k]
    for j in 1:model.n_breakpoints
        for k in 1:model.n_breakpoints
            # = 1/σ² * Σ (∂ŷ/∂ψ[j]) * (∂ŷ/∂ψ[k])
            hess[n_beta + j, n_beta + k] = sum(dy_dψ[:, j] .* dy_dψ[:, k]) / σ2
        end
    end
    
    # Block 4: ∂²NLL/∂β[j]∂log_σ = -2/σ² * Σ residuals * (∂ŷ/∂β[j])
    for j in 1:n_beta
        hess[j, n_params] = -2.0 / σ2 * sum(residuals .* dy_dβ[:, j])
        hess[n_params, j] = hess[j, n_params]  # Symmetry
    end
    
    # Block 5: ∂²NLL/∂ψ[j]∂log_σ = -2/σ² * Σ residuals * (∂ŷ/∂ψ[j])
    for j in 1:model.n_breakpoints
        hess[n_beta + j, n_params] = -2.0 / σ2 * sum(residuals .* dy_dψ[:, j])
        hess[n_params, n_beta + j] = hess[n_beta + j, n_params]  # Symmetry
    end
    
    # Block 6: ∂²NLL/∂(log_σ)² = 2*sum_sq_residuals/σ²
    hess[n_params, n_params] = 2.0 * sum_sq_residuals / σ2
    
    return hess
end
