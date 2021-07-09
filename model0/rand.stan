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
  vector[P] mu_p ;
  vector[O] alpha ;
  vector[S] gamma ;
  real mu ;
  vector[3] beta ;
  vector<lower=0>[3] sigma ;
}
transformed parameters {
  vector[I] odd = mu_p[plot] + 
                  alpha[openness] + gamma[soil] +
                  beta[1]*nci + beta[2]*density + beta[3]*shore ;
}
model {
  pflower ~ bernoulli_logit(odd) ;
  alpha ~ normal(0, sigma[1]) ;
  gamma ~ normal(0, sigma[2]) ;
  mu_p ~ normal(mu, sigma[3]) ;
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
