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
  int<lower=1> S ; // number of soil categories
  int<lower=1> O ; // number of openness categories
  int<lower=1> P ; // number of plots
  int<lower=0> y[I] ; // presence of flower
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
  vector[3] beta ;
  vector[P] delta ;
  real<lower=0> sigma ;
}
transformed parameters {
  vector[2] mu ;
  vector[I] count[2] ;
  mu[1] = mu0 ;
  mu[2] = beta[1] + diff ;
  count[1] = mu[1] + delta[plot] +
             beta[1]*nci + beta[2]*density + beta[3]*shore ;
  count[2] = mu[2] + delta[plot] +
             beta[1]*nci + beta[2]*density + beta[3]*shore ;
}
model {
  for(i in 1:I) 
    y[i] ~ TwoPoissonMixture_lpmf(p, to_vector(count[1:2, i])) ;
  delta ~ normal(0, sigma) ;
  mu0 ~ normal(0, 1) ;
  diff ~ normal(0, 1) ;
  beta ~ normal(0, 1) ;
  sigma ~ lognormal(0, 1) ;
}
generated quantities {
  vector[I] log_lik ;
  vector[I] prediction ;
  for(i in 1:I){
    log_lik[i] = log_sum_exp(log1m(p) + 
                             poisson_log_lpmf(y[i] | count[1, i]),
                             log(p) + 
                             poisson_log_lpmf(y[i] | count[2, i])) ;
    prediction[i] = TwoPoissonMixture_rng(p, to_vector(count[1:2, i])) ;  
  }
}
