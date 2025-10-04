"""
    print_results(results::FitResults; confidence_level=0.95)

For printing output to readable format.

"""


function print_results(results::FitResults; confidence_level=0.95)
    println("\n" * "="^70)
    println("MAP ESTIMATION RESULTS")
    println("="^70)
    println("Converged: ", Optim.converged(results.optim_result))
    println("Iterations: ", Optim.iterations(results.optim_result))
    println("Negative Log-likelihood: ", results.optim_result.minimum)
    println()
    
    # β parameters
    println("SLOPE/INTERCEPT PARAMETERS (β):")
    println("-" ^70)
    param_names = ["Intercept", "Slope 1"]
    for i in 3:length(results.model.β)
        push!(param_names, "Slope $(i-1)")
    end
    
    for i in 1:length(results.model.β)
        @printf("%-15s: %10.4f  (SE: %8.4f)  [%10.4f, %10.4f]\n",
                param_names[i],
                results.model.β[i],
                results.β_se[i],
                results.β_ci[i, 1],    # Fixed: was [1, i]
                results.β_ci[i, 2])    # Fixed: was [2, i]
    end
    
    # ψ parameters
    println("\nBREAKPOINT LOCATIONS (ψ):")
    println("-" ^70)
    for i in 1:length(results.model.ψ)
        ψ_continuous = (results.ψ_ci[i, 1] + results.ψ_ci[i, 2]) / 2    # Fixed
        @printf("ψ[%d]           : %10.2f  (SE: %8.4f)  [%10.2f, %10.2f]\n",
                i,
                ψ_continuous,
                results.ψ_se[i],
                results.ψ_ci[i, 1],    # Fixed: was [1, i]
                results.ψ_ci[i, 2])    # Fixed: was [2, i]
        @printf("  (rounded)    : %d\n", results.model.ψ[i])
    end
    
    # σ parameter
    println("\nNOISE PARAMETER (σ):")
    println("-" ^70)
    @printf("σ              : %10.4f  (SE: %8.4f)  [%10.4f, %10.4f]\n",
            results.model.σ,
            results.σ_se,
            results.σ_ci[1],
            results.σ_ci[2])
    
    # Correlation matrix
    println("\nCORRELATION MATRIX (selected parameters):")
    println("-" ^70)
    n_show = min(5, size(results.correlation_matrix, 1))
    for i in 1:n_show
        for j in 1:n_show
            @printf("%7.3f ", results.correlation_matrix[i, j])
        end
        println()
    end
    
    println("\n" * "="^70)
    @printf("Confidence Level: %.1f%%\n", confidence_level * 100)
    println("="^70)
end
