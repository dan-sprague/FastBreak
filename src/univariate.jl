"""
    predict(x, θ, n_breakpoints)

Evaluate piecewise linear model at points `x` using parameter vector `θ`.

Parameter vector structure:
θ = [β₁, β₂, ..., β_{n_breakpoints+2}, ψ₁, ..., ψ_{n_breakpoints}, log(σ)]
"""
function predict(x::Vector{Float64}, θ::Vector{Float64}, n_breakpoints::Int)
    n_beta = n_breakpoints + 2
    
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

"""
    nll(θ, model)

Negative log posterior for parameter vector θ and model.
"""
function nll(θ::Vector{Float64}, model::SegmentedModel)
    n = length(model.y)
    n_breakpoints = model.n_breakpoints
    n_beta = n_breakpoints + 2
    
    β = θ[1:n_beta]
    ψ = θ[n_beta+1:n_beta+n_breakpoints]
    log_σ = θ[end]
    σ = exp(log_σ)
    
    # Check if breakpoints are within prior bounds (for safety)
    ψ_min, ψ_max = model.ψ_prior_range
    if any(ψ .< ψ_min) || any(ψ .> ψ_max)
        return Inf  # Outside valid range
    end
    
    ŷ = predict(model.x, θ, n_breakpoints)
    
    residuals = model.y .- ŷ
    sum_sq_residuals = sum(residuals.^2)
    
    # Negative log likelihood
    nll_value = n * log_σ + sum_sq_residuals / (2 * σ^2)
    
    # Prior on β (slopes and intercept)
    prior_β_slopes = sum(β[2:end].^2) / (2 * model.slope_prior.σ^2)
    prior_β_intercept = (β[1] - model.intercept_prior.μ)^2 / (2 * model.intercept_prior.σ^2)
    
    # Prior on ψ (uniform over range - constant, doesn't affect optimization)
    # -log p(ψ) = log(ψ_max - ψ_min) for each breakpoint (constant)
    
    # Prior on σ (transform from prior on σ to prior on log_σ)
    # p(log_σ) = p(σ) * σ where σ = exp(log_σ)
    # For Exponential(λ): p(σ) = λ exp(-λσ)
    # -log p(log_σ) = -log(λ) - log(σ) + λσ = -log(λ) - log_σ + λ exp(log_σ)
    prior_σ = -logpdf(model.σ_prior, σ) - log_σ  # Jacobian adjustment
    
    return nll_value + prior_β_slopes + prior_β_intercept + prior_σ
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
              max_iter::Int=1000,
              show_trace::Bool=false,
              confidence_level::Float64=0.95)
    
    n_breakpoints = model.n_breakpoints
    n_beta = n_breakpoints + 2
    n_params = n_beta + n_breakpoints + 1
    
    # Initialize parameters
    θ_init = initialize_params(model)
    
    # Set up bounds for constrained optimization
    ψ_min, ψ_max = model.ψ_prior_range
    lower = vcat(
        fill(-Inf, n_beta),              # β unbounded
        fill(ψ_min, n_breakpoints),      # ψ within prior range
        -Inf                              # log_σ unbounded (but σ > 0)
    )
    upper = vcat(
        fill(Inf, n_beta),
        fill(ψ_max, n_breakpoints),
        Inf
    )
    
    # Define objective and gradient
    obj(θ) = nll(θ, model)
    
    function grad!(g, θ)
        gradient!(g, θ, model)
    end
    
    # Optimize with bounds (MAP estimation)
    println("Optimizing with analytical gradient (MAP estimation with bounds)...")
    result = optimize(obj, grad!, lower, upper, θ_init, Fminbox(LBFGS()),
                     Optim.Options(iterations=max_iter,
                                  show_trace=show_trace))
    
    θ_opt = Optim.minimizer(result) 
    
    # Compute Hessian at MAP estimate 
    println("Computing analytical Hessian at MAP estimate...")
    hess = zeros(n_params, n_params)
    hessian!(hess, θ_opt, model)
    
    # Approximate posterior covariance = inverse Hessian
    posterior_info = hess
    
    # Check positive definiteness
    eigvals_hess = eigvals(posterior_info)
    if any(eigvals_hess .<= 1e-10)
        @error "Hessian is not positive definite. Using regularization."
        println("Min eigenvalue: ", minimum(eigvals_hess))
        posterior_info = posterior_info + I * 1e-6
    end
    
    # Covariance matrix (approximate posterior covariance)
    covariance_matrix = inv(posterior_info)
    
    # Standard errors
    se = sqrt.(abs.(diag(covariance_matrix)))
    
    # Confidence intervals
    z = quantile(Normal(0, 1), 0.5 + confidence_level / 2)
    
    β_se = se[1:n_beta]
    β_ci = hcat(θ_opt[1:n_beta] .- z .* β_se, θ_opt[1:n_beta] .+ z .* β_se)
    
    ψ_se = se[n_beta+1:n_beta+n_breakpoints]
    ψ_ci = hcat(θ_opt[n_beta+1:n_beta+n_breakpoints] .- z .* ψ_se,
                θ_opt[n_beta+1:n_beta+n_breakpoints] .+ z .* ψ_se)
    
    # For log_σ, transform back to σ scale using delta method
    log_σ_opt = θ_opt[end]
    log_σ_se = se[end]
    σ_opt = exp(log_σ_opt)
    σ_se = σ_opt * log_σ_se  # Delta method
    σ_ci = (σ_opt - z * σ_se, σ_opt + z * σ_se)
    
    # Correlation matrix
    D_inv = Diagonal(1.0 ./ se)
    correlation_matrix = D_inv * covariance_matrix * D_inv
    
    fitted_params = FittedParams(
        Int.(round.(sort(θ_opt[n_beta+1:n_beta+n_breakpoints]))),
        θ_opt[1:n_beta],
        σ_opt
    )
    
    return FitResults(
        fitted_params,
        β_se,
        β_ci,
        ψ_se,
        ψ_ci,
        σ_se,
        σ_ci,
        covariance_matrix,
        correlation_matrix,
        hess,
        result
    )
end

"""
    predict(model::FittedParams, x)

Predict using fitted model.
"""
function predict(fitted::FittedParams, x::Vector{<:Real})
    n_breakpoints = length(fitted.ψ)
    θ = vcat(fitted.β, Float64.(fitted.ψ), log(fitted.σ))
    return predict(Float64.(x), θ, n_breakpoints)
end