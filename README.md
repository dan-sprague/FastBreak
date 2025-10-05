# FastBreak.jl

Fast, bayesian detection of breakpoints in univariate time series data. Envisioned for biologists interested in asking: what is the probability that population A and B reached plateau growth phase at the same time, controlling for growth rate? Statistical inference can also be performed on slopes (growth rates)

## Use Case

### Biology 
Identifying the probable interval of time that population growth curves change behavior (exponential/linear/plateau) can be calculated in fractions of a second. Hypothesis testing on breakpoints can be performed using the Wald test. 95% CI are reported for the slopes and breakpoint locations.

<p align="center">
  <img src="https://raw.githubusercontent.com/dan-sprague/FastBreak/main/img/population_growth_mcmc.png" alt="Population Growth 1" width="90%"/>
</p>

### Complicated functions 

However, more complex functions with closely spaced breakpoints require more complex fitting. FastBreak utilizes its hardcoded gradient and the `AdvancedHMC.jl` library to perform ultrafast Bayesian inference. FastBreak fits about 3X faster than an equivalent Stan model. 

<p align = "center">
<img src="https://raw.githubusercontent.com/dan-sprague/FastBreak/main/img/sine_map_vs_mcmc.png" alt="Population Growth 1" width="90%"/>
</p>


## Method Overview

FastBreak fits piecewise linear regression models with an arbitrary number of breakpoints using a fast MCMC sampler (< 1s for 2000 samples on hundreds of observations). All parameters are estimated **jointly** rather than in an iterative fashion. The package uses a hardcoded logposterior gradient to provide posterior samples in the blink of an eye. An initial MAP estimate is performed with Newton's method (Hessian derived with the assistance of Claude, because, well, no thanks!). 

### Why Bayes?

Standard optimization (MLE/MAP) performed well on simple growth curves, however that approach fails catastrophically on sine-like and more complex data, likely due to vanishing gradients and kinks in the loss induced by the breakpoints. R's `segmented` library uses an iterative fitting procedure however the hope for this library was for joint optimization of all parameters using gradient decent.

Bayesian sampling appears to better explore the solution space, leading optimal solutions even though the gradient optimization is still there.

## Features

Full Bayesian posterior for inference on generated and derived statistics!

- **Multiple breakpoints**: Fit models with any number of breakpoints
- **Fast optimization**: Analytical gradients and Hessians for Newton's method
- **Statistical inference**: Standard errors, confidence intervals, and correlation matrices on breakpoints and slopes. 
- **Simple API**: Intuitive interface for model specification and fitting


## Installation

```julia
using Pkg
Pkg.add("FastBreak")
```

## Model

FastBreak fits piecewise linear models of the form:

```
y = β₁ + β₂x + Σᵢ βᵢ₊₂ max(0, x - ψᵢ) + ε
```

where:
- `β₁` is the intercept
- `β₂` is the initial slope
- `βᵢ₊₂` are slope changes at each breakpoint
- `ψᵢ` are the breakpoint locations
- `ε ~ N(0, σ²)` is Gaussian noise

## Usage

### Basic Example

```julia
using FastBreak

# Generate synthetic data
x = collect(0.0:100.0)
true_breakpoints = [31, 69]
true_β = [10.0, 0.5, -1.0, 0.8]
true_σ = 3.0

# Create model
model = SegmentedModel(x, y, n_breakpoints=2)

# Fit model
results = fit!(model, show_trace=false)

# Print results
print_results(results)

======================================================================
MAXIMUM LIKELIHOOD ESTIMATION RESULTS
======================================================================
Converged: true
Iterations: 28
Negative Log-likelihood: 155.70509373999525

SLOPE/INTERCEPT PARAMETERS (β):
----------------------------------------------------------------------
Intercept      :    11.3248  (SE:   0.9509)  [    9.4610,    13.1887]
Slope 1        :     0.4184  (SE:   0.0495)  [    0.3213,     0.5155]
Slope 2        :    -1.0367  (SE:   0.0701)  [   -1.1740,    -0.8994]
Slope 3        :     0.9388  (SE:   0.0717)  [    0.7983,     1.0793]

BREAKPOINT LOCATIONS (ψ):
----------------------------------------------------------------------
ψ[1]           :      33.09  (SE:   1.3267)  [     30.49,      35.69]
  (rounded)    : 33
ψ[2]           :      67.53  (SE:   1.4757)  [     64.64,      70.42]
  (rounded)    : 68

NOISE PARAMETER (σ):
----------------------------------------------------------------------
σ              :     2.8338  (SE:   0.1994)  [    2.4430,     3.2246]

CORRELATION MATRIX (selected parameters):
----------------------------------------------------------------------
  1.000  -0.860   0.608   0.000   0.333 
 -0.860   1.000  -0.707  -0.000  -0.598 
  0.608  -0.707   1.000  -0.489  -0.021 
  0.000  -0.000  -0.489   1.000   0.433 
  0.333  -0.598  -0.021   0.433   1.000 

======================================================================
Confidence Level: 95.0%
======================================================================

# Make predictions
y_pred = results(x)
```

### Model Options

```julia
model = SegmentedModel(
    x, y,
    n_breakpoints;
    ψ_prior_range=(min_x, max_x)  # Bounds for breakpoint locations
)
```

### Fitting Options

```julia
results = fit!(
    model;
    max_iter=1000,           # Maximum optimization iterations
    show_trace=false,        # Show optimization progress
    confidence_level=0.95    # Confidence level for intervals
)
```

### Accessing Results

```julia
# Parameter estimates
results.θ.β      # Slope coefficients
results.θ.ψ      # Breakpoint locations
results.θ.σ      # Noise standard deviation

# Statistical inference
results.β_se     # Standard errors for β
results.β_ci     # Confidence intervals for β
results.ψ_se     # Standard errors for ψ
results.ψ_ci     # Confidence intervals for ψ

# Covariance structure
results.covariance_matrix
results.correlation_matrix
results.hessian
```

## Dependencies

- Optim.jl: Optimization algorithms
- LinearAlgebra: Matrix operations
- StatsBase: Statistical utilities

## License

See LICENSE file for details.

## Author

Dan Sprague (dsprague@broadinstitute.org)
