
"""
    hessian!(hess, θ, model)

Compute Hessian of negative log posterior with respect to θ in-place.
"""
function hessian!(hess::Matrix{Float64}, θ::Vector{Float64}, model::SegmentedModel)
    n = length(model.y)
    n_breakpoints = model.n_breakpoints
    n_beta = n_breakpoints + 2
    n_params = length(θ)
    
    β = θ[1:n_beta]
    ψ = θ[n_beta+1:n_beta+n_breakpoints]
    log_σ = θ[end]
    σ = exp(log_σ)
    σ2 = σ^2
    
    ŷ = predict(model.x, θ, n_breakpoints)
    residuals = model.y .- ŷ
    sum_sq_residuals = sum(residuals.^2)
    
    # Precompute Jacobian matrices
    dy_dβ = zeros(n, n_beta)
    dy_dβ[:, 1] .= 1.0
    dy_dβ[:, 2] .= model.x
    for i in 1:n_breakpoints
        dy_dβ[:, i+2] .= max.(0, model.x .- ψ[i])
    end
    
    dy_dψ = zeros(n, n_breakpoints)
    for i in 1:n_breakpoints
        indicators = Float64.(model.x .> ψ[i])
        dy_dψ[:, i] .= -β[i+2] .* indicators
    end
    
    fill!(hess, 0.0)
    
    # Block 1: ∂²/∂β[j]∂β[k]
    for j in 1:n_beta
        for k in 1:n_beta
            hess[j, k] = sum(dy_dβ[:, j] .* dy_dβ[:, k]) / σ2
        end
    end
    
    # Add prior contributions to diagonal
    hess[1, 1] += 1.0 / model.intercept_prior.σ^2
    for j in 2:n_beta
        hess[j, j] += 1.0 / model.slope_prior.σ^2
    end
    
    # Block 2: ∂²/∂β[j]∂ψ[k]
    for j in 1:n_beta
        for k in 1:n_breakpoints
            hess[j, n_beta + k] = sum(dy_dβ[:, j] .* dy_dψ[:, k]) / σ2
            
            if j == k + 2
                indicators = Float64.(model.x .> ψ[k])
                hess[j, n_beta + k] += sum(residuals .* (-indicators)) / σ2
            end
            
            hess[n_beta + k, j] = hess[j, n_beta + k]
        end
    end
    
    # Block 3: ∂²/∂ψ[j]∂ψ[k] (uniform prior contributes 0)
    for j in 1:n_breakpoints
        for k in 1:n_breakpoints
            hess[n_beta + j, n_beta + k] = sum(dy_dψ[:, j] .* dy_dψ[:, k]) / σ2
        end
    end
    
    # Block 4: ∂²/∂β[j]∂log_σ
    for j in 1:n_beta
        hess[j, n_params] = -2.0 / σ2 * sum(residuals .* dy_dβ[:, j])
        hess[n_params, j] = hess[j, n_params]
    end
    
    # Block 5: ∂²/∂ψ[j]∂log_σ
    for j in 1:n_breakpoints
        hess[n_beta + j, n_params] = -2.0 / σ2 * sum(residuals .* dy_dψ[:, j])
        hess[n_params, n_beta + j] = hess[n_beta + j, n_params]
    end
    
    # Block 6: ∂²/∂(log_σ)²
    hess_likelihood = 2.0 * sum_sq_residuals / σ2


    if model.σ_prior isa Exponential
        θ_prior = mean(model.σ_prior)  # scale parameter
        hess_prior = σ / θ_prior
    else
        hess_prior = 0.0  # Approximate for other distributions
    end

    hess[n_params, n_params] = hess_likelihood + hess_prior
    
    return hess
end
