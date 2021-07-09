functions {
  real TwoPoissonMixture_lpmf(int y, real prob, vector count) {
    real log_pdf[2];
    log_pdf[1] = log1m(prob) + poisson_log_lpmf(y | count[1]) ;
    log_pdf[2] = log(prob) + poisson_log_lpmf(y | count[2]) ;
    return log_sum_exp(log_pdf);
  }
  real TwoPoissonMixture_rng(real prob, vector count) {
    int z;
    z = bernoulli_rng(prob);
    return  z ? poisson_log_rng(count[2]) : poisson_log_rng(count[1]);
  }
}
data {
  int<lower=1> I ; // number of individuals
  int<lower=1> P ; // number of plots
  int<lower=0> y[I] ; // number of flowers
  vector[I] height ; // tree height
  vector[I] nci ; // mean distance to the three closest trees
  vector[I] density ; // plot tree density
  int<lower=1, upper=P> plot[I] ; // plots
}
parameters {
  real eta ;
  real gamma ;
  real mu0 ; 
  real<lower = 0> diff ;
  vector[2] beta ;
  vector[P] delta ;
  real<lower=0> sigma ;
}
transformed parameters {
  vector[2] mu ;
  vector[I] count[2] ;
  vector[I] p = inv_logit(rep_vector(eta, I) + gamma*density) ;
  mu[1] = mu0 ;
  mu[2] = mu[1] + diff ;
  count[1] = mu[1] + delta[plot] + beta[1]*height + beta[2]*nci ;
  count[2] = mu[2] + delta[plot] + beta[1]*height + beta[2]*nci ;
}
model {
  for(i in 1:I) 
    y[i] ~ TwoPoissonMixture_lpmf(p[i], to_vector(count[1:2, i])) ;
  delta ~ normal(0, sigma) ;
  eta ~ normal(0, 1) ;
  gamma ~ normal(0, 1) ;
  mu0 ~ normal(0, 1) ;
  diff ~ normal(0, 1) ;
  beta ~ normal(0, 1) ;
  sigma ~ lognormal(0, 1) ;
}
generated quantities {
  vector[I] log_lik ;
  vector[I] prediction ;
  for(i in 1:I){
    log_lik[i] = log_sum_exp(log1m(p[i]) + 
                             poisson_log_lpmf(y[i] | count[1, i]),
                             log(p[i]) + 
                             poisson_log_lpmf(y[i] | count[2, i])) ;
    prediction[i] = TwoPoissonMixture_rng(p[i], to_vector(count[1:2, i])) ;  
  }
}
