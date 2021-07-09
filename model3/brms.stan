functions {
 /* compute correlated group-level effects
  * Args: 
  *   z: matrix of unscaled group-level effects
  *   SD: vector of standard deviation parameters
  *   L: cholesky factor correlation matrix
  * Returns: 
  *   matrix of scaled group-level effects
  */ 
  matrix scale_r_cor(matrix z, vector SD, matrix L) {
    // r is stored in another dimension order than z
    return transpose(diag_pre_multiply(SD, L) * z);
  }
  /* zero-inflated poisson log-PDF of a single response 
   * Args: 
   *   y: the response value 
   *   lambda: mean parameter of the poisson distribution
   *   zi: zero-inflation probability
   * Returns:  
   *   a scalar to be added to the log posterior 
   */ 
  real zero_inflated_poisson_lpmf(int y, real lambda, real zi) { 
    if (y == 0) { 
      return log_sum_exp(bernoulli_lpmf(1 | zi), 
                         bernoulli_lpmf(0 | zi) + 
                         poisson_lpmf(0 | lambda)); 
    } else { 
      return bernoulli_lpmf(0 | zi) +  
             poisson_lpmf(y | lambda); 
    } 
  }
  /* zero-inflated poisson log-PDF of a single response 
   * logit parameterization of the zero-inflation part
   * Args: 
   *   y: the response value 
   *   lambda: mean parameter of the poisson distribution
   *   zi: linear predictor for zero-inflation part 
   * Returns:  
   *   a scalar to be added to the log posterior 
   */ 
  real zero_inflated_poisson_logit_lpmf(int y, real lambda, real zi) { 
    if (y == 0) { 
      return log_sum_exp(bernoulli_logit_lpmf(1 | zi), 
                         bernoulli_logit_lpmf(0 | zi) + 
                         poisson_lpmf(0 | lambda)); 
    } else { 
      return bernoulli_logit_lpmf(0 | zi) +  
             poisson_lpmf(y | lambda); 
    } 
  }
  /* zero-inflated poisson log-PDF of a single response
   * log parameterization for the poisson part
   * Args: 
   *   y: the response value 
   *   eta: linear predictor for poisson distribution
   *   zi: zero-inflation probability
   * Returns:  
   *   a scalar to be added to the log posterior 
   */ 
  real zero_inflated_poisson_log_lpmf(int y, real eta, real zi) { 
    if (y == 0) { 
      return log_sum_exp(bernoulli_lpmf(1 | zi), 
                         bernoulli_lpmf(0 | zi) + 
                         poisson_log_lpmf(0 | eta)); 
    } else { 
      return bernoulli_lpmf(0 | zi) +  
             poisson_log_lpmf(y | eta); 
    } 
  }
  /* zero-inflated poisson log-PDF of a single response 
   * log parameterization for the poisson part
   * logit parameterization of the zero-inflation part
   * Args: 
   *   y: the response value 
   *   eta: linear predictor for poisson distribution
   *   zi: linear predictor for zero-inflation part 
   * Returns:  
   *   a scalar to be added to the log posterior 
   */ 
  real zero_inflated_poisson_log_logit_lpmf(int y, real eta, real zi) { 
    if (y == 0) { 
      return log_sum_exp(bernoulli_logit_lpmf(1 | zi), 
                         bernoulli_logit_lpmf(0 | zi) + 
                         poisson_log_lpmf(0 | eta)); 
    } else { 
      return bernoulli_logit_lpmf(0 | zi) +  
             poisson_log_lpmf(y | eta); 
    } 
  }
  // zero-inflated poisson log-CCDF and log-CDF functions
  real zero_inflated_poisson_lccdf(int y, real lambda, real zi) { 
    return bernoulli_lpmf(0 | zi) + poisson_lccdf(y | lambda); 
  }
  real zero_inflated_poisson_lcdf(int y, real lambda, real zi) { 
    return log1m_exp(zero_inflated_poisson_lccdf(y | lambda, zi));
  }
}
data {
  int<lower=1> N;  // total number of observations
  int<lower=1> N_logheight;  // number of observations
  vector[N_logheight] Y_logheight;  // response variable
  int<lower=1> K_logheight;  // number of population-level effects
  matrix[N_logheight, K_logheight] X_logheight;  // population-level design matrix
  int<lower=1> N_logdiameter;  // number of observations
  vector[N_logdiameter] Y_logdiameter;  // response variable
  int<lower=1> K_logdiameter;  // number of population-level effects
  matrix[N_logdiameter, K_logdiameter] X_logdiameter;  // population-level design matrix
  int<lower=1> N_branches;  // number of observations
  int Y_branches[N_branches];  // response variable
  int<lower=1> K_branches;  // number of population-level effects
  matrix[N_branches, K_branches] X_branches;  // population-level design matrix
  int<lower=1> N_fruits;  // number of observations
  int Y_fruits[N_fruits];  // response variable
  int<lower=1> K_fruits;  // number of population-level effects
  matrix[N_fruits, K_fruits] X_fruits;  // population-level design matrix
  int<lower=1> N_flowers;  // number of observations
  int Y_flowers[N_flowers];  // response variable
  int<lower=1> K_flowers;  // number of population-level effects
  matrix[N_flowers, K_flowers] X_flowers;  // population-level design matrix
  // data for group-level effects of ID 1
  int<lower=1> N_1;  // number of grouping levels
  int<lower=1> M_1;  // number of coefficients per level
  int<lower=1> J_1_logheight[N_logheight];  // grouping indicator per observation
  int<lower=1> J_1_logdiameter[N_logdiameter];  // grouping indicator per observation
  int<lower=1> J_1_branches[N_branches];  // grouping indicator per observation
  int<lower=1> J_1_fruits[N_fruits];  // grouping indicator per observation
  int<lower=1> J_1_flowers[N_flowers];  // grouping indicator per observation
  // group-level predictor values
  vector[N_logheight] Z_1_logheight_1;
  vector[N_logdiameter] Z_1_logdiameter_2;
  vector[N_branches] Z_1_branches_3;
  vector[N_fruits] Z_1_fruits_4;
  vector[N_flowers] Z_1_flowers_5;
  int<lower=1> NC_1;  // number of group-level correlations
  int prior_only;  // should the likelihood be ignored?
}
transformed data {
  int Kc_logheight = K_logheight - 1;
  matrix[N_logheight, Kc_logheight] Xc_logheight;  // centered version of X_logheight without an intercept
  vector[Kc_logheight] means_X_logheight;  // column means of X_logheight before centering
  int Kc_logdiameter = K_logdiameter - 1;
  matrix[N_logdiameter, Kc_logdiameter] Xc_logdiameter;  // centered version of X_logdiameter without an intercept
  vector[Kc_logdiameter] means_X_logdiameter;  // column means of X_logdiameter before centering
  int Kc_branches = K_branches - 1;
  matrix[N_branches, Kc_branches] Xc_branches;  // centered version of X_branches without an intercept
  vector[Kc_branches] means_X_branches;  // column means of X_branches before centering
  int Kc_fruits = K_fruits - 1;
  matrix[N_fruits, Kc_fruits] Xc_fruits;  // centered version of X_fruits without an intercept
  vector[Kc_fruits] means_X_fruits;  // column means of X_fruits before centering
  int Kc_flowers = K_flowers - 1;
  matrix[N_flowers, Kc_flowers] Xc_flowers;  // centered version of X_flowers without an intercept
  vector[Kc_flowers] means_X_flowers;  // column means of X_flowers before centering
  for (i in 2:K_logheight) {
    means_X_logheight[i - 1] = mean(X_logheight[, i]);
    Xc_logheight[, i - 1] = X_logheight[, i] - means_X_logheight[i - 1];
  }
  for (i in 2:K_logdiameter) {
    means_X_logdiameter[i - 1] = mean(X_logdiameter[, i]);
    Xc_logdiameter[, i - 1] = X_logdiameter[, i] - means_X_logdiameter[i - 1];
  }
  for (i in 2:K_branches) {
    means_X_branches[i - 1] = mean(X_branches[, i]);
    Xc_branches[, i - 1] = X_branches[, i] - means_X_branches[i - 1];
  }
  for (i in 2:K_fruits) {
    means_X_fruits[i - 1] = mean(X_fruits[, i]);
    Xc_fruits[, i - 1] = X_fruits[, i] - means_X_fruits[i - 1];
  }
  for (i in 2:K_flowers) {
    means_X_flowers[i - 1] = mean(X_flowers[, i]);
    Xc_flowers[, i - 1] = X_flowers[, i] - means_X_flowers[i - 1];
  }
}
parameters {
  vector[Kc_logheight] b_logheight;  // population-level effects
  real Intercept_logheight;  // temporary intercept for centered predictors
  real<lower=0> sigma_logheight;  // residual SD
  vector[Kc_logdiameter] b_logdiameter;  // population-level effects
  real Intercept_logdiameter;  // temporary intercept for centered predictors
  real<lower=0> sigma_logdiameter;  // residual SD
  vector[Kc_branches] b_branches;  // population-level effects
  real Intercept_branches;  // temporary intercept for centered predictors
  vector[Kc_fruits] b_fruits;  // population-level effects
  real Intercept_fruits;  // temporary intercept for centered predictors
  real<lower=0,upper=1> zi_fruits;  // zero-inflation probability
  vector[Kc_flowers] b_flowers;  // population-level effects
  real Intercept_flowers;  // temporary intercept for centered predictors
  real<lower=0,upper=1> zi_flowers;  // zero-inflation probability
  vector<lower=0>[M_1] sd_1;  // group-level standard deviations
  matrix[M_1, N_1] z_1;  // standardized group-level effects
  cholesky_factor_corr[M_1] L_1;  // cholesky factor of correlation matrix
}
transformed parameters {
  matrix[N_1, M_1] r_1;  // actual group-level effects
  // using vectors speeds up indexing in loops
  vector[N_1] r_1_logheight_1;
  vector[N_1] r_1_logdiameter_2;
  vector[N_1] r_1_branches_3;
  vector[N_1] r_1_fruits_4;
  vector[N_1] r_1_flowers_5;
  // compute actual group-level effects
  r_1 = scale_r_cor(z_1, sd_1, L_1);
  r_1_logheight_1 = r_1[, 1];
  r_1_logdiameter_2 = r_1[, 2];
  r_1_branches_3 = r_1[, 3];
  r_1_fruits_4 = r_1[, 4];
  r_1_flowers_5 = r_1[, 5];
}
model {
  // likelihood including constants
  if (!prior_only) {
    // initialize linear predictor term
    vector[N_logheight] mu_logheight = Intercept_logheight + rep_vector(0.0, N_logheight);
    // initialize linear predictor term
    vector[N_logdiameter] mu_logdiameter = Intercept_logdiameter + rep_vector(0.0, N_logdiameter);
    // initialize linear predictor term
    vector[N_branches] mu_branches = Intercept_branches + rep_vector(0.0, N_branches);
    // initialize linear predictor term
    vector[N_fruits] mu_fruits = Intercept_fruits + Xc_fruits * b_fruits;
    // initialize linear predictor term
    vector[N_flowers] mu_flowers = Intercept_flowers + Xc_flowers * b_flowers;
    for (n in 1:N_logheight) {
      // add more terms to the linear predictor
      mu_logheight[n] += r_1_logheight_1[J_1_logheight[n]] * Z_1_logheight_1[n];
    }
    for (n in 1:N_logdiameter) {
      // add more terms to the linear predictor
      mu_logdiameter[n] += r_1_logdiameter_2[J_1_logdiameter[n]] * Z_1_logdiameter_2[n];
    }
    for (n in 1:N_branches) {
      // add more terms to the linear predictor
      mu_branches[n] += r_1_branches_3[J_1_branches[n]] * Z_1_branches_3[n];
    }
    for (n in 1:N_fruits) {
      // add more terms to the linear predictor
      mu_fruits[n] += r_1_fruits_4[J_1_fruits[n]] * Z_1_fruits_4[n];
    }
    for (n in 1:N_flowers) {
      // add more terms to the linear predictor
      mu_flowers[n] += r_1_flowers_5[J_1_flowers[n]] * Z_1_flowers_5[n];
    }
    target += normal_id_glm_lpdf(Y_logheight | Xc_logheight, mu_logheight, b_logheight, sigma_logheight);
    target += normal_id_glm_lpdf(Y_logdiameter | Xc_logdiameter, mu_logdiameter, b_logdiameter, sigma_logdiameter);
    target += poisson_log_glm_lpmf(Y_branches | Xc_branches, mu_branches, b_branches);
    for (n in 1:N_fruits) {
      target += zero_inflated_poisson_log_lpmf(Y_fruits[n] | mu_fruits[n], zi_fruits);
    }
    for (n in 1:N_flowers) {
      target += zero_inflated_poisson_log_lpmf(Y_flowers[n] | mu_flowers[n], zi_flowers);
    }
  }
  // priors including constants
  target += student_t_lpdf(Intercept_logheight | 3, 0.9, 2.5);
  target += student_t_lpdf(sigma_logheight | 3, 0, 2.5)
    - 1 * student_t_lccdf(0 | 3, 0, 2.5);
  target += student_t_lpdf(Intercept_logdiameter | 3, 3.1, 2.5);
  target += student_t_lpdf(sigma_logdiameter | 3, 0, 2.5)
    - 1 * student_t_lccdf(0 | 3, 0, 2.5);
  target += student_t_lpdf(Intercept_branches | 3, 3.3, 2.5);
  target += student_t_lpdf(Intercept_fruits | 3, -2.3, 2.5);
  target += beta_lpdf(zi_fruits | 1, 1);
  target += student_t_lpdf(Intercept_flowers | 3, -2.3, 2.5);
  target += beta_lpdf(zi_flowers | 1, 1);
  target += student_t_lpdf(sd_1 | 3, 0, 2.5)
    - 5 * student_t_lccdf(0 | 3, 0, 2.5);
  target += std_normal_lpdf(to_vector(z_1));
  target += lkj_corr_cholesky_lpdf(L_1 | 1);
}
generated quantities {
  // actual population-level intercept
  real b_logheight_Intercept = Intercept_logheight - dot_product(means_X_logheight, b_logheight);
  // actual population-level intercept
  real b_logdiameter_Intercept = Intercept_logdiameter - dot_product(means_X_logdiameter, b_logdiameter);
  // actual population-level intercept
  real b_branches_Intercept = Intercept_branches - dot_product(means_X_branches, b_branches);
  // actual population-level intercept
  real b_fruits_Intercept = Intercept_fruits - dot_product(means_X_fruits, b_fruits);
  // actual population-level intercept
  real b_flowers_Intercept = Intercept_flowers - dot_product(means_X_flowers, b_flowers);
  // compute group-level correlations
  corr_matrix[M_1] Cor_1 = multiply_lower_tri_self_transpose(L_1);
  vector<lower=-1,upper=1>[NC_1] cor_1;
  // extract upper diagonal of correlation matrix
  for (k in 1:M_1) {
    for (j in 1:(k - 1)) {
      cor_1[choose(k - 1, 2) + j] = Cor_1[j, k];
    }
  }
}
