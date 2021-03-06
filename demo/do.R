library(pgbart)

f = function(x){
  10*sin(pi*x[,1]*x[,2]) + 20*(x[,3]-.5)^2+10*x[,4]+5*x[,5]
}

sigma = 2.0  #y = f(x) + sigma*z , z~N(0,1)
n = 100      #number of observations
set.seed(99)
x=matrix(runif(n*10),n,10) #10 variables, only first 5 matter
Ey = f(x)
y=Ey+sigma*rnorm(n)

fit.pgbart <- pgbart_train(train_data = x,
                           train_label = y,
                           model_file = "./pgbart.model",
                           nskip=1, ndpost=20,
                           m_bart = 100,
                           if_test = FALSE,
                           verbose_level = 0,
                           if_set_seed = FALSE,
                           mcmc_type = "cgm") #the output path is locale by default.
summary(fit.pgbart$train$sigma)
