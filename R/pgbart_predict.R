pgbart_predict <- function(test_data, model_file) {
	if(is.null(test_data))
    	stop("ERROR: you miss a test_data parameter!")
  	if(!is.matrix(test_data))
    	stop("ERROR: test_data must be a matrix!")
    if(is.null(model_file))
   		stop("ERROR: you miss a model_file parameter!")
  	if(!is.character(model_file))
    	stop("ERROR: model_file must be a character!")
    if(!file.exists(model_file))
    	stop("ERROR: model_file does not exist!")

    pred_result <- predict(test_data, model_file)
    val <- list(yhat=NULL, yhat.mean=NULL)
    val$yhat <- pred_result
    val$yhat.mean <- colMeans(pred_result)
    val
}
