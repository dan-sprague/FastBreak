# ======================= NOTE ===================================
# This hessian was derived by CLAUDE because of its complexity.
# It has been verified against autodiff.
# ================================================================



"""
    hessian!(hess, θ, model)

Compute Hessian of negative log posterior with respect to θ in-place.
"""
function hessian!(hess, θ, model::SegmentedModel)
    n = length(model.y)
    n_beta = model.n_breakpoints + 2
    n_params = length(θ)

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

    # Compute Hessian in the ordered space first
    # Precompute derivatives of prediction w.r.t. parameters
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

    # Get prior parameters
    mean_y = mean(model.y)
    sd_y = std(model.y)

    # Compute Hessian in ordered space
    hess_ordered = zeros(n_params, n_params)

    # Block 1: ∂²NLL/∂β[j]∂β[k] = 1/σ² * Σ (∂ŷ/∂β[j]) * (∂ŷ/∂β[k])
    for j in 1:n_beta
        for k in 1:n_beta
            hess_ordered[j, k] = sum(dy_dβ[:, j] .* dy_dβ[:, k]) / σ2
        end
    end

    # Add prior contributions to β block (diagonal only, priors are independent)
    # beta[1] ~ normal(mean_y, sd_y*2): ∂²/∂β[1]² = 1/(sd_y*2)^2
    prior_var_intercept = (sd_y * 2)^2
    hess_ordered[1, 1] += 1.0 / prior_var_intercept

    # beta[2:(K+2)] ~ normal(0, 10): ∂²/∂β[i]² = 1/100
    for i in 2:n_beta
        hess_ordered[i, i] += 1.0 / 100.0
    end

    # Block 2: ∂²NLL/∂β[j]∂ψ[k]
    for j in 1:n_beta
        for k in 1:model.n_breakpoints
            hess_ordered[j, n_beta + k] = sum(dy_dβ[:, j] .* dy_dψ[:, k]) / σ2

            if j == k + 2
                indicators = Float64.(x .> ψ[k])
                hess_ordered[j, n_beta + k] += sum(residuals .* (-indicators)) / σ2
            end

            hess_ordered[n_beta + k, j] = hess_ordered[j, n_beta + k]
        end
    end

    # Block 3: ∂²NLL/∂ψ[j]∂ψ[k]
    for j in 1:model.n_breakpoints
        for k in 1:model.n_breakpoints
            hess_ordered[n_beta + j, n_beta + k] = sum(dy_dψ[:, j] .* dy_dψ[:, k]) / σ2
        end
    end

    # Block 4: ∂²NLL/∂β[j]∂log_σ
    for j in 1:n_beta
        hess_ordered[j, n_params] = -2.0 / σ2 * sum(residuals .* dy_dβ[:, j])
        hess_ordered[n_params, j] = hess_ordered[j, n_params]
    end

    # Block 5: ∂²NLL/∂ψ[j]∂log_σ
    for j in 1:model.n_breakpoints
        hess_ordered[n_beta + j, n_params] = -2.0 / σ2 * sum(residuals .* dy_dψ[:, j])
        hess_ordered[n_params, n_beta + j] = hess_ordered[n_beta + j, n_params]
    end

    # Block 6: ∂²NLL/∂(log_σ)²
    # Likelihood contribution
    hess_ordered[n_params, n_params] = 2.0 * sum_sq_residuals / σ2
    # Prior contribution: sigma ~ exponential(10), ∂²/∂(log_σ)² = 10*σ
    hess_ordered[n_params, n_params] += 10.0 * σ

    # Now transform Hessian to unconstrained space
    # Build Jacobian matrix J where J[i,j] = ∂ψ[i]/∂θ_ψ[j]
    J = zeros(model.n_breakpoints, model.n_breakpoints)
    for i in 1:model.n_breakpoints
        for j in 1:model.n_breakpoints
            if j == 1
                J[i, j] = 1.0  # ∂ψ[i]/∂θ_ψ[1] = 1 for all i
            elseif j <= i
                J[i, j] = exp(θ_ψ[j])  # ∂ψ[i]/∂θ_ψ[j] = exp(θ_ψ[j]) for j <= i
            end
        end
    end

    # Copy β and σ blocks (unchanged)
    hess[1:n_beta, 1:n_beta] .= hess_ordered[1:n_beta, 1:n_beta]
    hess[n_params, n_params] = hess_ordered[n_params, n_params]

    # Transform ψ-ψ block: H_θψ = J^T * H_ψ * J + second order terms
    H_ψ = hess_ordered[n_beta+1:n_beta+model.n_breakpoints, n_beta+1:n_beta+model.n_breakpoints]

    # Initialize ψ-ψ block to zeros
    hess[n_beta+1:n_beta+model.n_breakpoints, n_beta+1:n_beta+model.n_breakpoints] .= 0.0

    # Compute gradient w.r.t. ψ for second order terms
    grad_ψ = Vector{Float64}(undef, model.n_breakpoints)
    for i in 1:model.n_breakpoints
        indicators = Float64.(x .> ψ[i])
        grad_ψ[i] = β[i+2] / σ2 * sum(residuals .* indicators)
    end

    # Second order term: ∂²ψ[i]/∂θ_ψ[j]²
    for i in 2:model.n_breakpoints
        for j in 2:model.n_breakpoints
            if i >= j
                # Add Σ_k (∂L/∂ψ[k]) * (∂²ψ[k]/∂θ_ψ[i]∂θ_ψ[j])
                # ∂²ψ[k]/∂θ_ψ[i]² = exp(θ_ψ[i]) for k >= i
                if i == j
                    hess[n_beta + i, n_beta + j] = sum(grad_ψ[i:end]) * exp(θ_ψ[i])
                end
            end
        end
    end

    # Add first order term: J^T * H_ψ * J
    hess[n_beta+1:n_beta+model.n_breakpoints, n_beta+1:n_beta+model.n_breakpoints] .+= J' * H_ψ * J

    # Transform β-ψ blocks
    for j in 1:n_beta
        H_βψ = hess_ordered[j, n_beta+1:n_beta+model.n_breakpoints]
        # H_βψ is a row vector, so J' * H_βψ gives the transformed column
        transformed = J' * H_βψ
        hess[n_beta+1:n_beta+model.n_breakpoints, j] .= transformed
        hess[j, n_beta+1:n_beta+model.n_breakpoints] .= transformed
    end

    # Transform ψ-σ blocks
    H_ψσ = hess_ordered[n_beta+1:n_beta+model.n_breakpoints, n_params]
    transformed_ψσ = J' * H_ψσ
    hess[n_beta+1:n_beta+model.n_breakpoints, n_params] .= transformed_ψσ
    hess[n_params, n_beta+1:n_beta+model.n_breakpoints] .= transformed_ψσ

    # Copy β-σ blocks
    hess[1:n_beta, n_params] .= hess_ordered[1:n_beta, n_params]
    hess[n_params, 1:n_beta] .= hess_ordered[n_params, 1:n_beta]

    return hess
end
