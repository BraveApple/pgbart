pgbart_train <- function(train_data, train_label, if_test = FALSE, test_data = matrix(nrow = 0, ncol = 0), test_label = vector(length = 0),
  model_file, alpha_bart = 3.0, q_bart = 0.9, alpha_split = 0.95, beta_split = 0.5, if_center_label = FALSE,
	ess_threshold = 1.0, init_seed_id = 1, if_set_seed = TRUE,k_bart = 2.0, m_bart = 200, min_size = 1, ndpost = 200, nskip = 100, keepevery = 1,
	verbose_level = 1, n_particles = 10) {
  
  if_debug=FALSE
  variance_type = "unconditional"
  resample_type = "multinomial"
  if(is.null(train_data))
    stop("ERROR: you miss a train_data parameter!")
  if(!is.matrix(train_data))
    stop("ERROR: train_data must be a matrix!")
  if(is.null(train_label))
    stop("ERROR: you miss a train_label parameter!")
  if(!is.vector(train_label))
    stop("ERROR: train_label must be a vector!")
  if(nrow(train_data)!= length(train_label))
    stop("ERROR: the number of train_data must be equal to the length of train_label!")
  if(if_test) {
    if(is.null(test_data))
      stop("ERROR: you miss a test_data parameter!")
    if(!is.matrix(test_data))
      stop("ERROR: test_data must be a matrix!")
    if(is.null(test_label))
      stop("ERROR: you miss a test_label parameter!")
    if(!is.vector(test_label))
      stop("ERROR: test_label must be a vector!")
    if(dim(test_data)[1] != length(test_label))
      stop("ERROR: the number of train_data must be equal to the length of train_label!")
  }
  if(is.null(model_file))
    stop("ERROR: you miss a model_file parameter!")
  if(!is.character(model_file))
    stop("ERROR: model_file must be a character!")
  if(file.exists(model_file))
    file.remove(model_file)
  if(alpha_bart < 0)
    stop("alpha_bart is the df parameter in BART, it must be positive, default 3.0!")
  if(!(q_bart >= 0 && q_bart <= 1.0))
    stop("q_bart controls the prior over sigma^2 in BART, and needs to be in [0, 1], default 0.9!")
  if(!(alpha_split >= 0 && alpha_split <= 1.0))
    stop("alpha_split for cgm tree prior and needs to be in [0, 1], default 0.95!")
  if(beta_split < 0)
    stop("(1/beta_split) for cgm tree prior and needs to be greater than 0, default 0.5!")
  if(ess_threshold < 0)
    stop("ess_threshold needs to be in [0, 1], default 1.0!")
  if(init_seed_id < 0)
    stop("ess_threshold needs to be greater than 0, default 1!")
  if(k_bart < 0)
    stop("k_bart controls the prior over mu (mu_prec) in BART, it must be positive, default 2.0!")
  if(m_bart < 1)
    stop("m_bart specifies the number of trees in BART, it must be not less than 1, default 1!")
  if(min_size < 1)
    stop("min_size is minimum number of data points at leaf nodes, it must be not less than 1, default 1!")
  if(ndpost < 1)
    stop("ERROR: ndpost must be not less than 1")
  if(nskip < 1)
    stop("ERROR: nskip must be not less than 1")
  if(keepevery < 1)
    stop("ERROR: keepevery must be not less than 1")
  if(variance_type != "unconditional")
    stop("variance_type must be \"unconditional\"!")
  if(verbose_level < 0 || verbose_level >= 2)
    stop("verbosity level (0 is minimum, 1 is maximum), default 1!")
  if(n_particles < 1)
    stop("n_particles is the number of particles, it must be not less than 1, default 10!")
  if(resample_type != "multinomial" && resample_type != "systematic")
    stop("control.resample_type must be \"multinomial\" or \"systematic\"!")


  val_tmp <- train(train_data, train_label, if_test, test_data, test_label, model_file, alpha_bart, alpha_split, beta_split, if_center_label,
    if_debug, ess_threshold, init_seed_id, if_set_seed, k_bart, m_bart, min_size, ndpost,
    nskip, keepevery, variance_type, q_bart, verbose_level, n_particles, resample_type)

  if(if_center_label){
    ori_mean <- mean(train_label)
    val_tmp$yhat.train <- val_tmp$yhat.train + ori_mean
    val_tmp$yhat.test <- val_tmp$yhat.test + ori_mean
  }

  val_tmp$yhat.train.mean <- colMeans(val_tmp$yhat.train)
  val_tmp$yhat.test.mean <- colMeans(val_tmp$yhat.test)
  if (!if_test) {
    val_tmp$yhat.test <- NULL
    val_tmp$mse.test <- NULL
    val_tmp$loglik.test <- NULL
    val_tmp$yhat.test.mean <- NULL
  }
  val <- list(train=NULL, test=NULL)
  val$train <- list(yhat=val_tmp$yhat.train, yhat.mean=val_tmp$yhat.train.mean ,mse=val_tmp$mse.train, loglik=val_tmp$loglik.train,
    first.sigma = 1 / sqrt(val_tmp$first.sigma), sigma= 1 / sqrt(val_tmp$sigma), varcount=val_tmp$varcount)
  val$test <- list(yhat=val_tmp$yhat.test, yhat.mean=val_tmp$yhat.test.mean, mse=val_tmp$mse.test, loglik=val_tmp$loglik.test)
  val
}

