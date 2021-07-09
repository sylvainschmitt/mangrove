data {
  int<lower=1> I ; // number of individuals
  int<lower=1> S ; // number of soil categories
  int<lower=1> O ; // number of openness categories
  int<lower=1> P ; // number of plots
  int<lower=0, upper=1> pflower[I] ; // presence of flower
  vector[I] nci ; // mean distance to the three closest trees
  vector[I] density ; // plot tree density
  vector[I] shore ; // distance from shore
  vector[I] soil ;
  vector[I] openness ;
  int<lower=1, upper=P> plot[I] ;
}
parameters {
  vector[P] mu_p ;
  real mu ;
  vector[5] beta ;
  real<lower=0> sigma ;
}
transformed parameters {
  vector[I] odd = mu_p[plot] + 
                  beta[1]*nci + beta[2]*density + beta[3]*shore +
                  beta[4]*openness + beta[5]*soil ;
}
model {
  pflower ~ bernoulli_logit(odd) ;
  mu_p ~ normal(mu, sigma) ;
  mu ~ normal(0, 1) ;
  beta ~ normal(0, 1) ;
  sigma ~ lognormal(0, 1) ;
}
generated quantities {
  vector[I] log_lik ;
  vector[I] prediction ;
  for(i in 1:I){
    log_lik[i] = bernoulli_logit_lpmf(pflower[i] | odd[i]) ;
    prediction[i] = bernoulli_logit_rng(odd[i]) ;  
  }
}
