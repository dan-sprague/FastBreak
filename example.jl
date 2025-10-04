using FastBreak
function generate_predictions(x::Vector{Float64}, θ::Vector{Float64}, n_breakpoints::Int)
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

function example()
    # Generate synthetic data
    x = collect(0.0:100.0)
    true_breakpoints = [30, 70]
    true_β = [10.0, 0.5, -1.0, 0.8]
    true_σ = 3.0
    
    n_breakpoints = length(true_breakpoints)
    θ_true = vcat(true_β, Float64.(true_breakpoints), log(true_σ))

    y_true = generate_predictions(x, θ_true, n_breakpoints)
    y_noisy = y_true .+ randn(length(x)) * true_σ
    
    # Create model with data and priors
    model = SegmentedModel(
        x, y_noisy, 
        n_breakpoints
    )
    
    # Fit model
    results = fit!(model, show_trace=false)
    
    return results
end

