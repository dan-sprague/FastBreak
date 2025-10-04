data {
  int<lower=0> N;                   // Number of data points
  int<lower=1> K;       // Number of breakpoints
  vector[N] x;                      // Time post inoculation
  vector[N] y;                      // Prion titer
}

transformed data {
  real min_x = min(x);
  real max_x = max(x);
  real mean_y = mean(y);
  real sd_y = sd(y);
}

parameters {
  real<lower=0> sigma;                      // Error term
  vector[K + 2] beta;           // Regression coefficients
  ordered[K] psi;               // ordered breakpoints (psi_1 < psi_2 < ...)
}

model {
  // Linear predictor
  vector[N] mu;

  // Priors for the model parameters
  sigma ~ exponential(10);
  beta[1] ~ normal(mean_y, sd_y * 2);      // Prior for the intercept beta_0


  // Priors for all slope-related coefficients
  beta[2:(K + 2)] ~ normal(0, 10);
  psi ~ uniform(min_x, max_x);             // Priors for the breakpoints


  
  mu = beta[1] + beta[2] * x; // initial growth phase
  
  // Each new breakpoint slope represents the change from the previous slope!!!!
  for (k in 1:K) {
    mu += beta[k + 2] * fmax(0, x - psi[k]);
  }
  
  // Likelihood
  y ~ normal(mu, sigma);
}