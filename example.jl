using FastBreak,Distributions

function example()
    # Generate synthetic data
    x = collect(0.0:100.0)
    true_breakpoints = [30, 70]
    true_β = [10.0, 0.5, -1.0, 0.8]
    true_σ = 3.0
    
    n_breakpoints = length(true_breakpoints)
    θ_true = vcat(true_β, Float64.(true_breakpoints), log(true_σ))
    
    y_true = predict(x, θ_true, n_breakpoints)
    y_noisy = y_true .+ randn(length(x)) * true_σ
    
    # Create model with data and priors
    model = SegmentedModel(
        x, y_noisy, n_breakpoints,
        slope_prior=Normal(0, 100),
        intercept_prior=Normal(0, 100),
        σ_prior=Exponential(5.0),           # Prior on σ
        ψ_prior_range=(0.0, 100.0)          # Uniform prior on breakpoints
    )
    
    # Fit model
    results = fit!(model, show_trace=false)
    
    return results
end


[print_results(example()) for i in 1:100]


