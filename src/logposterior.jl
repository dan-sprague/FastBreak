"""
    transform_to_ordered(θ_unconstrained)

Transform unconstrained breakpoint parameters to ordered breakpoints.
Uses STAN's ordered parameter transformation:
- ψ[1] = θ[1] (unconstrained)
- ψ[i] = ψ[i-1] + exp(θ[i]) for i > 1
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
function nll(θ, model::SegmentedModel)
    n = length(model.y)
    n_breakpoints = model.n_breakpoints
    n_beta = n_breakpoints + 2

    β = θ[1:n_beta]
    θ_ψ = θ[n_beta+1:n_beta+n_breakpoints]
    ψ = transform_to_ordered(θ_ψ)  # Transform to ordered space
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

    # Jacobian adjustment for ordered transformation
    # When sampling in unconstrained space θ_ψ with priors on constrained ψ,
    # we need: log p(θ_ψ) = log p(ψ(θ_ψ)) + log|det(∂ψ/∂θ_ψ)|
    # For ordered transform: log|det(J)| = sum(θ_ψ[2:end])
    # So we subtract this from the negative log posterior
    log_jacobian_det = n_breakpoints > 1 ? sum(θ_ψ[2:end]) : 0.0

    # MAP objective = negative log posterior (with Jacobian adjustment)
    return neg_log_likelihood + neg_log_prior - log_jacobian_det
end
"""
    fit!(model; kwargs...)

Fit maximum a posteriori (MAP) estimate for segmented regression model.

# Arguments
- `model::SegmentedModel`: model with data and priors
- `max_iter::Int`: maximum optimization iterations (default: 1000)
- `show_trace::Bool`: show optimization trace (default: false)
- `confidence_level::Float64`: confidence level for intervals (default: 0.95)
"""
function fit!(model::SegmentedModel;
                max_iter=1000,
                show_trace=false,
                confidence_level=0.95)
    
    x = model.x
    y = model.y
    n_breakpoints = model.n_breakpoints

    n_beta = n_breakpoints + 2
    n_params = n_beta + n_breakpoints + 1

    # Initialize
    β_init = randn(n_beta) * (std(y) / 10)
    β_init[1] = mean(y)

    # Initialize breakpoint locations
    ψ_init_ordered = if n_breakpoints == 1
        [quantile(x, 0.5)]
    else
        [quantile(x, q) for q in range(0.1, 0.9, length=n_breakpoints)]
    end

    # Transform to unconstrained space
    θ_ψ_init = transform_from_ordered(ψ_init_ordered)
    log_σ_init = log(std(y))

    params_init = vcat(β_init, θ_ψ_init, log_σ_init)

    # Define objective and gradient
    obj(p) = nll(p, model)

    function grad!(g, p)
        gradient!(g, p, model)
    end

    function hess!(h, p)
        hessian!(h, p, model)
    end

    println("Optimizing...")
    result = optimize(obj, grad!, hess!, params_init, Newton(),
                     Optim.Options(iterations=max_iter,
                                  show_trace=show_trace))

    params_opt = Optim.minimizer(result)
    
    # Compute Hessian at MLE
    println("Computing analytical Hessian...")
    hess = zeros(n_params, n_params)
    hessian!(hess, params_opt, model)
    
    # Fisher Information = Hessian (for MLE)
    fisher_info = hess
    
    # Check positive definiteness
    eigvals_hess = eigvals(fisher_info)
    if any(eigvals_hess .<= 1e-10)
        @warn "Hessian is not positive definite. Using regularization."
        println("Min eigenvalue: ", minimum(eigvals_hess))
        fisher_info = fisher_info + I * 1e-6
    end
    
    covariance_matrix = inv(fisher_info)
    
    z_crit = 1.96  # For ~95% CI

    # Extract parameters
    β_opt = params_opt[1:n_beta]
    θ_ψ_opt = params_opt[n_beta+1:n_beta+n_breakpoints]
    ψ_opt_continuous = transform_to_ordered(θ_ψ_opt)
    log_σ_opt = params_opt[end]
    σ_opt = exp(log_σ_opt)
    
    # Standard errors for β (no transformation needed)
    β_se = sqrt.(abs.(diag(covariance_matrix[1:n_beta, 1:n_beta])))
    
    # Transform ψ covariance using Jacobian (delta method)
    cov_θ_ψ = covariance_matrix[n_beta+1:n_beta+n_breakpoints, 
                                 n_beta+1:n_beta+n_breakpoints]
    
    # Build Jacobian: J[i,j] = ∂ψ[i]/∂θ_ψ[j]
    J = zeros(n_breakpoints, n_breakpoints)
    for i in 1:n_breakpoints
        for j in 1:n_breakpoints
            if j == 1
                J[i, j] = 1.0  # All ψ depend on θ_ψ[1]
            elseif j <= i
                J[i, j] = exp(θ_ψ_opt[j])  # ψ[i] depends on θ_ψ[j] for j ≤ i
            end
        end
    end
    
    # Transform covariance: Cov(ψ) = J * Cov(θ_ψ) * J^T
    cov_ψ = J * cov_θ_ψ * J'
    ψ_se = sqrt.(abs.(diag(cov_ψ)))  # Correct standard errors in ψ space
    
    # Standard error for log_σ
    log_σ_se = sqrt(abs(covariance_matrix[end, end]))
    
    # Confidence intervals
    β_ci = zeros(2, n_beta)
    β_ci[1, :] = β_opt .- z_crit .* β_se
    β_ci[2, :] = β_opt .+ z_crit .* β_se
    
    ψ_ci = zeros(2, n_breakpoints)
    ψ_ci[1, :] = ψ_opt_continuous .- z_crit .* ψ_se  # Now correct!
    ψ_ci[2, :] = ψ_opt_continuous .+ z_crit .* ψ_se
    
    # σ using delta method
    σ_se = σ_opt * log_σ_se
    σ_ci_lower = σ_opt - z_crit * σ_se
    σ_ci_upper = σ_opt + z_crit * σ_se
    
    # Compute full standard errors and correlation matrix
    standard_errors = vcat(β_se, ψ_se, σ_se)
    
    # Build full covariance matrix in transformed space
    cov_full = zeros(n_params, n_params)
    cov_full[1:n_beta, 1:n_beta] = covariance_matrix[1:n_beta, 1:n_beta]
    cov_full[n_beta+1:n_beta+n_breakpoints, n_beta+1:n_beta+n_breakpoints] = cov_ψ
    cov_full[end, end] = covariance_matrix[end, end]
    
    # Transform cross-covariances
    # Cov(β, ψ) = Cov(β, θ_ψ) * J^T
    cov_full[1:n_beta, n_beta+1:n_beta+n_breakpoints] = 
        covariance_matrix[1:n_beta, n_beta+1:n_beta+n_breakpoints] * J'
    cov_full[n_beta+1:n_beta+n_breakpoints, 1:n_beta] = 
        cov_full[1:n_beta, n_beta+1:n_beta+n_breakpoints]'
    
    # Cov(ψ, σ) = J * Cov(θ_ψ, log_σ)
    cov_full[n_beta+1:n_beta+n_breakpoints, end] = 
        J * covariance_matrix[n_beta+1:n_beta+n_breakpoints, end]
    cov_full[end, n_beta+1:n_beta+n_breakpoints] = 
        cov_full[n_beta+1:n_beta+n_breakpoints, end]'
    
    # Cov(β, σ) unchanged
    cov_full[1:n_beta, end] = covariance_matrix[1:n_beta, end]
    cov_full[end, 1:n_beta] = covariance_matrix[end, 1:n_beta]
    
    D_inv = Diagonal(1.0 ./ standard_errors)
    correlation_matrix = D_inv * cov_full * D_inv

    # Create fitted parameters with continuous breakpoint estimates
    θ_mle = FittedParams(ψ_opt_continuous, β_opt, σ_opt)

    return FittedSegmentModel(
        θ_mle,
        β_se,
        β_ci,
        ψ_se,
        ψ_ci,
        σ_se,
        (σ_ci_lower, σ_ci_upper),
        cov_full,  # Use transformed covariance
        correlation_matrix,
        hess,
        result
    )
end