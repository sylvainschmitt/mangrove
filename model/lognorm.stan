data {
  int<lower=1> N ; // number of observations
  int<lower=1> S ; // number of soil categories
  int<lower=1> O ; // number of openness categories
  int<lower=1> P ; // number of plots
  vector[N] trait ; // response variable (height, diameter, branches, fruits, or flowers)
  vector[N] nci ; // mean distance to the three closest trees
  vector[N] density ; // plot tree density
  vector[N] shore ; // distance from shore
  int<lower=1, upper=S> soil[N] ;
  int<lower=1, upper=O> openness[N] ;
  int<lower=1, upper=P> plot[N] ;
}
parameters {
  real<lower=0> alpha ;
  real beta_nci ;
  real beta_density ;
  real beta_shore ;
  vector[S] delta_soil ;
  vector[O] delta_openness ;
  vector[P] gamma_plot ;
  real<lower=0> sigma ;
  real<lower=0> sigma_plot ;
}
model {
  trait ~ lognormal(log(alpha + 
                        beta_nci*nci + beta_density*density + beta_shore*shore + 
                        delta_soil[soil] +  delta_openness[openness] +
                        gamma_plot[plot]),
                    sigma) ;
  gamma_plot ~ normal(0, sigma_plot) ;
  delta_soil ~ normal(0, 1) ;
  delta_openness ~ normal(0, 1) ;
}
generated quantities {
  real Vnci = variance(beta_nci*nci) ;
  real Vdensity = variance(beta_density*density) ;
  real Vshore = variance(beta_shore*shore) ;
  real Vsoil = variance(beta_shore*shore) ;
  real Vopenness = variance(beta_shore*shore) ;
  real Vplot = variance(beta_shore*shore) ;
  vector[N] residuals = trait - (alpha + beta_nci*nci + beta_density*density + beta_shore*shore + 
                                 delta_soil[soil] +  delta_openness[openness] + gamma_plot[plot]) ;
  real Vresiduals = variance(residuals) ;
}
