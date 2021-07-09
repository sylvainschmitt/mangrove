data {
  int<lower=1> I ; // number of individuals
  int<lower=1> P ; // number of plots
  int<lower=0> y[I] ; // number of flowers
  vector[I] height ; // tree height
  vector[I] nci ; // mean distance to the three closest trees
  int<lower=1, upper=P> plot[I] ; // plots
}
parameters {
  real mu ; 
  vector[2] beta ;
  vector[P] mu_p ;
  real<lower=0> sigma ;
}
transformed parameters {
    vector[I] count = mu_p[plot] + beta[1]*height + beta[2]*nci ;
}
model {
  y ~ poisson_log(count) ;
  mu_p ~ normal(mu, sigma) ;
  mu ~ normal(0, 1) ;
  beta ~ normal(0, 1) ;
  sigma ~ lognormal(0, 1) ;
}
generated quantities {
  vector[I] log_lik ;
  vector[I] prediction ;
  for(i in 1:I){
    log_lik[i] = poisson_log_lpmf(y[i] | count[i]) ;
    prediction[i] = poisson_log_rng(count[i]) ;  
  }
}
