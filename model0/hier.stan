functions {
  real TwoBernoulliMixture_lpmf(int y, real prob, vector odd) {
    real log_pdf[2];
    log_pdf[1] = log1m(prob) + bernoulli_logit_lpmf(y | odd[1]) ;
    log_pdf[2] = log(prob) + bernoulli_logit_lpmf(y | odd[2]) ;
    return log_sum_exp(log_pdf);
  }
  real TwoBernoulliMixture_rng(real prob, vector odd) {
    int z;
    z = bernoulli_rng(prob);
    return  z ? bernoulli_logit_rng(odd[2]) : bernoulli_logit_rng(odd[1]);
  }
}
data {
  int<lower=1> I ; // number of individuals
  int<lower=1> S ; // number of soil categories
  int<lower=1> O ; // number of openness categories
  int<lower=1> P ; // number of plots
  int<lower=0, upper=1> pflower[I] ; // presence of flower
  vector[I] nci ; // mean distance to the three closest trees
  vector[I] density ; // plot tree density
  vector[I] shore ; // distance from shore
  int<lower=1, upper=S> soil[I] ;
  int<lower=1, upper=O> openness[I] ;
  int<lower=1, upper=P> plot[I] ;
}
parameters {
  real<lower = 0.0, upper = 1.0> p;
  real mu0 ; 
  real<lower = 0> diff ;
  // vector[2] mu ;
  vector[3] beta ;
  vector[P] delta ;
  real<lower=0> sigma ;
}
transformed parameters {
  vector[2] mu ;
  vector[I] odd[2] ;
  mu[1] = mu0 ;
  mu[2] = beta[1] + diff ;
  // real diff = abs(mu[1] - mu[2]) ;
  odd[1] = mu[1] + delta[plot] +
           beta[1]*nci + beta[2]*density + beta[3]*shore ;
  odd[2] = mu[2] + delta[plot] +
           beta[1]*nci + beta[2]*density + beta[3]*shore ;
}
model {
  for(i in 1:I) 
    pflower[i] ~ TwoBernoulliMixture_lpmf(p, to_vector(odd[1:2, i])) ;
  delta ~ normal(0, sigma) ;
  mu0 ~ normal(0, 1) ;
  diff ~ normal(0, 1) ;
  // mu ~ normal(0, 1) ;
  beta ~ normal(0, 1) ;
  sigma ~ lognormal(0, 1) ;
}
generated quantities {
  vector[I] log_lik ;
  vector[I] prediction ;
  for(i in 1:I){
    log_lik[i] = log_sum_exp(log1m(p) + 
                 bernoulli_logit_lpmf(pflower[i] | odd[1, i]), 
                 log(p) + 
                 bernoulli_logit_lpmf(pflower[i] | odd[2, i])) ;
    prediction[i] = TwoBernoulliMixture_rng(p, to_vector(odd[1:2, i])) ;  
  }
}
