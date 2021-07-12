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
  int<lower=0> flowers[I] ; // number of flowers
  int<lower=0> fruits[I] ; // number of fruits
  vector[I] height ; // tree height
  vector[I] nci ; // mean distance to the three closest trees
  int<lower=1, upper=P> plot[I] ; // plots
}
parameters {
  real<lower = 0.0, upper = 1.0> p;
  real mu0Fl ; 
  real mu0Fr ; 
  real<lower = 0> diffFl ;
  real<lower = 0> diffFr ;
  vector[2] betaFl ;
  vector[2] betaFr ;
  vector[P] deltaFl ;
  vector[P] deltaFr ;
  real<lower=0> sigmaFl ;
  real<lower=0> sigmaFr ;
}
transformed parameters {
  vector[2] muFl ;
  vector[2] muFr ;
  vector[I] countFl[2] ;
  vector[I] countFr[2] ;
  muFl[1] = mu0Fl ;
  muFr[1] = mu0Fr ;
  muFl[2] = muFl[1] + diffFl ;
  muFr[2] = muFr[1] + diffFr ;
  countFl[1] = muFl[1] + deltaFl[plot] + betaFl[1]*height + betaFl[2]*nci ;
  countFr[1] = muFr[1] + deltaFr[plot] + betaFr[1]*height + betaFr[2]*nci ;
  countFl[2] = muFl[2] + deltaFl[plot] + betaFl[1]*height + betaFl[2]*nci ;
  countFr[2] = muFr[2] + deltaFr[plot] + betaFr[1]*height + betaFr[2]*nci ;
}
model {
  for(i in 1:I) {
    flowers[i] ~ TwoPoissonMixture_lpmf(p, to_vector(countFl[1:2, i])) ;
    fruits[i] ~ TwoPoissonMixture_lpmf(p, to_vector(countFr[1:2, i])) ;
  }
  deltaFl ~ normal(0, sigmaFl) ;
  deltaFr ~ normal(0, sigmaFr) ;
  mu0Fl ~ normal(0, 1) ;
  mu0Fr ~ normal(0, 1) ;
  diffFl ~ normal(0, 1) ;
  diffFr ~ normal(0, 1) ;
  betaFl ~ normal(0, 1) ;
  betaFr ~ normal(0, 1) ;
  sigmaFl ~ lognormal(0, 1) ;
  sigmaFr ~ lognormal(0, 1) ;
}
generated quantities {
  vector[I] log_lik ;
  vector[I] predFl ;
  vector[I] predFr ;
  for(i in 1:I){
    log_lik[i] = log_sum_exp(log1m(p) + 
                             poisson_log_lpmf(flowers[i] | countFl[1, i]),
                             log(p) + 
                             poisson_log_lpmf(flowers[i] | countFl[2, i])) +
                  log_sum_exp(log1m(p) + 
                             poisson_log_lpmf(fruits[i] | countFr[1, i]),
                             log(p) + 
                             poisson_log_lpmf(fruits[i] | countFr[2, i])) ;
    predFl[i] = TwoPoissonMixture_rng(p, to_vector(countFl[1:2, i])) ;
    predFr[i] = TwoPoissonMixture_rng(p, to_vector(countFr[1:2, i])) ;  
  }
}
