functions {
  real TwoGaussianMixture_lpdf(real y, real prob, vector location, real scale) {
    real log_pdf[2];
    log_pdf[1] = log1m(prob) + normal_lpdf(y| location[1], scale);
    log_pdf[2] = log(prob) + normal_lpdf(y| location[2], scale);
    return log_sum_exp(log_pdf);
  }
  
  real TwoGaussianMixture_rng(real prob, vector location, real scale) {
    int z;
    z = bernoulli_rng(prob);
    return  z ? normal_rng(location[2], scale) : normal_rng(location[1], scale);
  }
}
data {
  int<lower=1> I ; // number of individuals
  int<lower=1> S ; // number of soil categories
  int<lower=1> O ; // number of openness categories
  int<lower=1> P ; // number of plots
  real y[I] ; // presence of flower
  vector[I] nci ; // mean distance to the three closest trees
  vector[I] density ; // plot tree density
  vector[I] shore ; // distance from shore
  int<lower=1, upper=S> soil[I] ;
  int<lower=1, upper=O> openness[I] ;
  int<lower=1, upper=P> plot[I] ;
}
parameters {
  real<lower = 0.0, upper = 1.0> p;
  real alpha0 ; 
  real<lower = 0> diff ;
  vector[3] beta ;
  vector[P] delta ;
  vector<lower=0>[2] sigma ;
}
transformed parameters {
  vector[2] alpha ;
  vector[I] mu[2] ;
  alpha[1] = alpha0 ;
  alpha[2] = beta[1] + diff ;
  mu[1] = alpha[1] + delta[plot] +
          beta[1]*nci + beta[2]*density + beta[3]*shore ;
  mu[2] = alpha[2] + delta[plot] +
          beta[1]*nci + beta[2]*density + beta[3]*shore ;
}
model {
  for(i in 1:I) 
    log(y[i]) ~ TwoGaussianMixture_lpdf(p, to_vector(mu[1:2, i]), sigma[1]) ;
  delta ~ normal(0, sigma[2]) ;
  alpha0 ~ normal(0, 1) ;
  diff ~ normal(0, 1) ;
  beta ~ normal(0, 1) ;
  sigma ~ lognormal(0, 1) ;
}
generated quantities {
  vector[I] log_lik ;
  vector[I] prediction ;
  for(i in 1:I){
    log_lik[i] = log_sum_exp(log1m(p) + 
                             normal_lpdf(log(y[i]) | mu[1, i], sigma[1]),
                             log(p) + 
                             normal_lpdf(log(y[i]) | mu[2, i], sigma[1])) ;
    prediction[i] = exp(TwoGaussianMixture_rng(p, to_vector(mu[1:2, i]), sigma[1])) ;  
  }
}
