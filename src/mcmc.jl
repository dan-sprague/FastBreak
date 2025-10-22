"""
    LogDensityGradient

Struct that computes both log density and gradient.
Implements LogDensityProblems.jl interface for AdvancedHMC.
"""
struct LogDensityGradient
    model::SegmentedModel
    n_params::Int
end

# Constructor that infers dimension
function LogDensityGradient(model::SegmentedModel)
    n_params = model.n_breakpoints + 2 + model.n_breakpoints + 1
    LogDensityGradient(model, n_params)
end

# Implement LogDensityProblems.jl interface
LogDensityProblems.dimension(ℓ::LogDensityGradient) = ℓ.n_params

LogDensityProblems.capabilities(::Type{<:LogDensityGradient}) = LogDensityProblems.LogDensityOrder{1}()

function LogDensityProblems.logdensity(ℓ::LogDensityGradient, θ::AbstractVector)
    return -negativeloglikelihood(θ, ℓ.model)
end

function LogDensityProblems.logdensity_and_gradient(ℓ::LogDensityGradient, θ::AbstractVector)
    # Compute log posterior
    log_p = -negativeloglikelihood(θ, ℓ.model)

    # Compute gradient
    ∇log_p = zeros(length(θ))
    gradient!(∇log_p, θ, ℓ.model)
    ∇log_p .*= -1  # Negate because gradient! computes gradient of -log posterior

    return log_p, ∇log_p
end

"""
    sample_logposterior(model::SegmentedModel;
                n_samples=1000,
                n_adapts=1000,
                init_params=nothing,
                seed=nothing)

Sample from the posterior using NUTS (No-U-Turn Sampler) with δ=0.8.

# Arguments
- `model::SegmentedModel`: the segmented regression model
- `n_samples::Int`: number of posterior samples (default: 1000)
- `n_adapts::Int`: number of adaptation steps (default: 1000)
- `init_params::Vector`: initial parameters (default: MAP estimate)
- `seed::Int`: random seed for reproducibility

# Returns
- `MCMCChains.Chains`: MCMC chain with samples

# Example
```julia
model = SegmentedModel(x, y, 4)
chain = sample_logposterior(model, n_samples=2000, n_adapts=1000)

# Extract breakpoint samples
psi_samples = chain[:psi]
```
"""
function sample_logposterior(model::SegmentedModel;
                     n_samples::Int=1000,
                     n_adapts::Int=1000,
                     δ = 0.65,
                     init_params::Union{Nothing,Vector{Float64}}=nothing,
                     seed::Union{Nothing,Int}=nothing)

    # Set random seed if provided
    rng = isnothing(seed) ? Random.default_rng() : Random.MersenneTwister(seed)

    n_breakpoints = model.n_breakpoints
    n_beta = n_breakpoints + 2
    n_params = n_beta + n_breakpoints + 1

    # Initialize parameters
    if isnothing(init_params)
        println("Computing MAP estimate for initialization...")
        map_result = fit!(model, show_trace=false)

        # Extract MAP parameters in unconstrained space
        θ_init = vcat(
            map_result.θ.β,
            transform_from_ordered(Float64.(map_result.θ.ψ)),
            log(map_result.θ.σ)
        )
    else
        θ_init = init_params
    end

    # Create log density object
    ℓ = LogDensityGradient(model)

    # Wrap in LogDensityModel for AbstractMCMC interface
    ℓ_model = AbstractMCMC.LogDensityModel(ℓ)
    nuts = NUTS(δ)

    # Run sampling with adaptation
    println("Running NUTS sampling...")
    println("  Adaptation: $n_adapts steps")
    println("  Sampling: $n_samples steps")

    samples_raw = sample(
        rng,
        ℓ_model,
        nuts,
        n_samples;
        n_adapts=n_adapts,
        initial_params=θ_init,
        progress=true
    )

    # Convert samples to array (n_samples × n_params)
    # Extract parameters from each sample using .z.θ
    samples_array = zeros(n_samples, n_params)
    for i in 1:n_samples
        samples_array[i, :] = samples_raw[i].z.θ
    end

    # Transform theta_psi to psi for each sample
    psi_samples = zeros(n_samples, n_breakpoints)
    for i in 1:n_samples
        θ_ψ = samples_array[i, n_beta+1:n_beta+n_breakpoints]
        psi_samples[i, :] = transform_to_ordered(θ_ψ)
    end

    # Transform log_sigma to sigma
    sigma_samples = exp.(samples_array[:, end])

    # Combine transformed samples
    transformed_samples = hcat(
        samples_array[:, 1:n_beta],  # beta
        psi_samples,                  # psi (transformed)
        sigma_samples                 # sigma (transformed)
    )

    transformed_names = vcat(
        ["beta[$i]" for i in 1:n_beta],
        ["psi[$i]" for i in 1:n_breakpoints],
        ["sigma"]
    )

    # Create MCMCChains object
    chain = Chains(
        transformed_samples,
        transformed_names,
        (parameters = transformed_names,)
    )

    # Print diagnostics
    println("\nSampling complete!")
    # Extract stats from samples
    acceptance_rates = [s.stat.acceptance_rate for s in samples_raw]
    step_sizes = [s.stat.step_size for s in samples_raw]
    println("  Mean acceptance rate: $(mean(acceptance_rates))")
    println("  Final step size: $(step_sizes[end])")

    return chain
end