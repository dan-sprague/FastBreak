"""
    predict(x, θ, n_breakpoints)

Evaluate piecewise linear model at points `x` using parameter vector `θ`.

Parameter vector structure:
θ = [β₁, β₂, ..., β_{n_breakpoints+2}, ψ₁, ..., ψ_{n_breakpoints}, log(σ)]
"""
function predict(model::SegmentedModel, θ::Vector{Float64})
    n_beta = model.n_breakpoints + 2
    n_breakpoints = model.n_breakpoints

    x = model.x
    # Extract parameters
    β = θ[1:n_beta]
    ψ = θ[n_beta+1:n_beta+n_breakpoints]
    
    # Compute predictions
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
    ψ = θ[n_beta+1:n_beta+n_breakpoints]
    log_σ = θ[end]
    σ = exp(log_σ)
    
    # Compute predictions
    y_pred = fill(β[1], n)
    y_pred .+= β[2] .* model.x
    for i in 1:n_breakpoints
        y_pred .+= β[i+2] .* max.(0, model.x .- ψ[i])
    end
    
    # NLL = n*log(σ) + 1/(2σ²)*Σ(residuals²)
    residuals = model.y .- y_pred
    sum_sq_residuals = sum(residuals.^2)
    
    nll = n * log_σ + sum_sq_residuals / (2 * σ^2)
    
    return nll
end

"""
    initialize_params(model)

Initialize parameter vector from priors and data.
"""
function initialize_params(model::SegmentedModel)
    n_breakpoints = model.n_breakpoints
    n_beta = n_breakpoints + 2
    
    β_init = Vector{Float64}(undef, n_beta)
    β_init[1] = mean(model.y)  
    β_init[2:end] .= rand(model.slope_prior, n_breakpoints + 1) .* 0.1
    
    # Initialize breakpoints within valid range
    ψ_min, ψ_max = model.ψ_prior_range
    ψ_init = [ψ_min + (ψ_max - ψ_min) * q 
              for q in range(0.2, 0.8, length=n_breakpoints)]
    
    # Initialize log_σ from prior
    log_σ_init = log(mean(model.σ_prior))
    
    return vcat(β_init, ψ_init, log_σ_init)
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
    β_init = randn(n_beta) * 0.1
    β_init[1] = mean(y)
    ψ_init = [quantile(x, q) for q in range(0.2, 0.8, length=n_breakpoints)]
    log_σ_init = log(std(y))
    
    params_init = vcat(β_init, ψ_init, log_σ_init)
    
    # Define objective and gradient
    obj(p) = nll(p, model)

    function grad!(g, p)
        gradient!(g, p, model)
    end

    function hess!(h, p)
        hessian!(h, p, model)
    end
    
    println("Optimizing...")
    result = optimize(obj, grad!,params_init, Newton(),
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
        # Add small diagonal regularization
        fisher_info = fisher_info + I * 1e-6
    end
    
    covariance_matrix = inv(fisher_info)
    
    standard_errors = sqrt.(abs.(diag(covariance_matrix)))
    
    D_inv = Diagonal(1.0 ./ standard_errors)
    correlation_matrix = D_inv * covariance_matrix * D_inv
    
    z_crit = 1.96  # Approx for 95% CI
    if confidence_level != 0.95
        throw(ArgumentError("Only 0.95 confidence level is currently supported teehee"))
    end

    # Extract parameters
    β_opt = params_opt[1:n_beta]
    ψ_opt_continuous = params_opt[n_beta+1:n_beta+n_breakpoints]
    log_σ_opt = params_opt[end]
    σ_opt = exp(log_σ_opt)
    
    # Standard errors
    β_se = standard_errors[1:n_beta]
    ψ_se = standard_errors[n_beta+1:n_beta+n_breakpoints]
    log_σ_se = standard_errors[end]
    
    # Confidence intervals
    β_ci = zeros(2, n_beta)
    β_ci[1, :] = β_opt .- z_crit .* β_se
    β_ci[2, :] = β_opt .+ z_crit .* β_se
    
    ψ_ci = zeros(2, n_breakpoints)
    ψ_ci[1, :] = ψ_opt_continuous .- z_crit .* ψ_se
    ψ_ci[2, :] = ψ_opt_continuous .+ z_crit .* ψ_se
    
    # σ using delta method
    σ_se = σ_opt * log_σ_se
    σ_ci_lower = σ_opt - z_crit * σ_se
    σ_ci_upper = σ_opt + z_crit * σ_se
    
    # Create model
    ψ_opt_int = round.(Int, sort(ψ_opt_continuous))
    θ_mle = FittedParams(ψ_opt_int, β_opt, σ_opt)

    return FittedSegmentModel(
        θ_mle,
        β_se,
        β_ci,
        ψ_se,
        ψ_ci,
        σ_se,
        (σ_ci_lower, σ_ci_upper),
        covariance_matrix,
        correlation_matrix,
        hess,
        result
    )
end
