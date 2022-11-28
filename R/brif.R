#' @title brif: A tree and forest tool for classification and regression 
#' 
#' @description Build decision trees and random forests for classification and regression. The implementation strikes a balance between minimizing computing efforts and maximizing the expected predictive accuracy, thus scales well to large data sets. Multi-threading is available through 'OpenMP'. 
#' 
#' @section Available functions:
#' Use \code{\link[brif]{brif}} to build a random forest and (optionally) make predictions.
#' Use \code{\link[brif]{brifTree}} to build a single decision tree.
#' Use \code{\link[brif]{printRules}} to print out the decision rules of a tree.
#' Use \code{\link[brif]{predict.brif}} to make predictions using a brif model (tree or forest).
#' 
#'
#' @author Yanchao Liu
#'
#' @docType package
#' @name brif-package
NULL


#' Build a model (and make predictions)
#' 
#' Depending on the arguments supplied, the function \code{\link[brif]{brif.formula}},  \code{\link[brif]{brif.default}} or \code{\link[brif]{brif.trainpredict}} will be called.
#' @param x a data frame or a \code{\link[stats]{formula}} object.
#' @param ... arguments passed on to \code{\link{brif.formula}}, \code{\link{brif.default}} or \code{\link{brif.trainpredict}}. 
#' @return a data frame, a vector or a list. If \code{newdata} is supplied,  prediction results for \code{newdata} will be returned in a data frame or a vector, depending on the problem type (classification or regression) and the \code{type} argument; otherwise, an object of class "brif" is returned, which is to be used in the function \code{\link[brif]{predict.brif}} for making predictions. See \code{\link[brif]{brif.default}} for components of the "brif" object. 
#'
#' @examples
#' trainset <- sample(1:nrow(iris), 0.5*nrow(iris))
#' validset <- setdiff(1:nrow(iris), trainset)
#'
#' # Train and predict at once 
#' pred_scores <- brif(Species~., data = iris, subset = trainset, 
#'                     newdata = iris[validset, 1:4], type = 'score')
#' pred_labels <- brif(Species~., data = iris, subset = trainset, 
#'                     newdata = iris[validset, 1:4], type = 'class')
#'
#' # Confusion matrix
#' table(pred_labels, iris[validset, 5])
#' 
#' # Accuracy
#' sum(pred_labels == iris[validset, 5])/length(validset)
#' 
#' # Train using the formula format
#' bf <- brif(Species~., data = iris, subset = trainset)
#' 
#' # Or equivalently, train using the data.frame format
#' bf <- brif(iris[trainset, c(5,1:4)])
#' 
#' # Make a prediction 
#' pred_scores <- predict(bf, iris[validset, 1:4], type = 'score')
#' pred_labels <- predict(bf, iris[validset, 1:4], type = 'class')
#' 
#' # Regression
#' bf <- brif(mpg ~., data = mtcars)
#' pred <- predict(bf, mtcars[2:11])
#' plot(pred, mtcars$mpg)
#' abline(0, 1)
#' 
#' # Optionally, delete the model object to release memory
#' rm(list = c("bf"))
#' gc()

#' @export
#' 
brif <- function(x, ...) UseMethod("brif");

#' Build a model (and make predictions) with formula
#'
#' @param formula an object of class "\code{\link[stats]{formula}}": a symbolic description of the model to be fitted.
#' @param data an optional data frame, list or environment (or object coercible by \code{\link{as.data.frame}} to a data frame) containing the variables in the model. If not found in \code{data}, the variables are taken from \code{environment(formula)}, typically the environment from which \code{brif.formula} is called.
#' @param subset an optional vector specifying a subset (in terms of index numbers, not actual data) of observations to be used in the fitting process.
#' @param na.action a function which indicates what should happen when the data contain NAs. 
#' @param newdata a data frame containing the data set for prediction. Default is NULL. If newdata is supplied, prediction results will be returned. 
#' @param type a character string specifying the prediction format, which takes effect only when \code{newdata} is supplied. Available values include "score" and "class". Default is "score". 
#' @param ... additional algorithmic parameters. See \code{\link[brif]{brif.default}} for a complete list. 
#' @return an object of class \code{brif} to be used by \code{\link{predict.brif}}. 
#' @examples
#' bf <- brif(Species ~ ., data = iris)
#' pred <- predict(bf, iris[,1:4])
#' @export
#'
brif.formula <- function(formula, data, subset, na.action = stats::na.pass, newdata=NULL, type = c("score","class"), ...) {
  Call <- match.call()
  indx <- match(c("formula", "data", "subset"), names(Call), nomatch = 0L)
  if(indx[1] == 0L) stop("a 'formula' argument is required")
  temp <- Call[c(1L, indx)]      # only keep the arguments we wanted
  temp$na.action <- na.action    # This one has a default
  temp[[1L]] <- quote(stats::model.frame) # change the function called
  mf <- eval.parent(temp)
  Terms <- attr(mf, "terms")
  indx <- match(c("newdata"), names(Call), nomatch = 0L)
  if(indx[1] == 0L){
    return(brif.default(mf[,c(attr(mf, "names")[1],attr(Terms, "term.labels"))], ...))
  } else {
    if(is.data.frame(newdata)){
      this_type <- match.arg(type)
      return(brif.trainpredict(mf[,c(attr(mf, "names")[1],attr(Terms, "term.labels"))], newdata, this_type, ...))
    } else {
      stop("newdata is provided but it is not a data frame.")
    }
  }
}

#' Stratified permutation of rows by the first column
#' 
#' @param x a data frame to be permuted by row
#' @param stride an integer indicating how many rows are to be groups in one block
#' 
#' @return a data frame, which is a permutation of x
#' 
stratpar <- function(x, stride){
  n <- nrow(x)
  if(n %% stride != 0) stop("Number of rows is not a multiple of stride.")
  ordermat <- matrix(1:n, byrow = F, ncol = stride)
  neworder <- as.vector(t(ordermat))
  x <- x[order(x[,1]),]  # sort by the first column
  x <- x[neworder,]
  return(x)
}

#' Build a model taking a data frame as input
#' 
#' @param x a data frame containing the training data set. The first column is taken as the target variable and all other columns are used as predictors. 
#' @param n_numeric_cuts an integer value indicating the maximum number of split points to generate for each numeric variable. 
#' @param n_integer_cuts an integer value indicating the maximum number of split points to generate for each integer variable. 
#' @param max_integer_classes an integer value. If the target variable is integer and has more than max_integer_classes unique values in the training data, then the target variable will be grouped into max_integer_classes bins. If the target variable is numeric, then the smaller of max_integer_classes and the number of unique values number of bins will be created on the target variables and the regression problem will be solved as a classification problem. 
#' @param max_depth an integer specifying the maximum depth of each tree. Default is 10. Maximum is 40. 
#' @param min_node_size an integer specifying the minimum number of training cases a leaf node must contain. Default is 1. 
#' @param ntrees an integer specifying the number of trees in the forest. 
#' @param seed an integer specifying the seed used by the internal random number generator. Default is 0, meaning not to set a seed but to accept the set seed from the calling environment.
#' @param ps an integer indicating the number of predictors to sample at each node split. Default is 0, meaning to use sqrt(p), where p is the number of predictors in the input. 
#' @param max_factor_levels an integer. If any factor variables has more than max_factor_levels, the program stops and prompts the user to increase the value of this parameter if the too-many-level factor is indeed intended. 
#' @param bagging_method an integer indicating the bagging sampling method: 0 for sampling without replacement; 1 for sampling with replacement (bootstrapping). 
#' @param bagging_proportion a numeric scalar between 0 and 1, indicating the proportion of training observations to be used in each tree. 
#' @param split_search an integer indicating the choice of the split search method. 0: randomly pick a split point (fastest, greatest variance, least effective); 1: do a greedy search starting from the mid-point; 2: evaluate every available split point and pick the best one; 3: perform a self-regulating local search to prevent over-fitting.
#' @param search_radius an positive integer indicating the split point search radius. This parameter takes effect only when split_search is 3. 
#' @param verbose an integer (0 or 1) specifying the verbose level.  
#' @param nthreads an integer specifying the number of threads used by the program. Default is 2. This parameter takes effect only on systems supporting OPENMP.  
#' @param ... additional arguments.
#' @return an object of class \code{brif}, which is a list containing the following components. Note: this object is not intended for any use other than that by the function \code{\link[brif]{predict.brif}}. Do not apply the \code{\link[utils]{str}} function on this object because the output can be long and meaningless especially when ntrees is large. Use \code{\link[base]{summary}} to get a peek of its structure. Use \code{\link[brif]{printRules}} to print out the decision rules of a particular tree. Most of the data in the object is stored in the tree_leaves element (which is a list of lists by itself) of this list. 
#' \item{p}{an integer scalar, the number of variables (predictors) used in the model}
#' \item{var_types}{an character vector of length (p+1) containing the variable names, including the target variable name as its first element}
#' \item{var_labels}{an character vector of length (p+1) containing the variable types, including that of the target variable as its first element}
#' \item{n_bcols}{an integer vector of length (p+1), containing the numbers of binary columns generated for each variable}
#' \item{ntrees}{an integer scalar indicating the number of trees in the model}
#' \item{index_in_group}{an integer vector specifying the internal index, for each variable, in its type group}
#' \item{numeric_cuts}{a list containing split point information on numeric variables}
#' \item{integer_cuts}{a list containing split point information on integer variables}
#' \item{factor_cuts}{a list containing split point information on factor variables}
#' \item{n_num_vars}{an integer scalar indicating the numeric variables in the model}
#' \item{n_int_vars}{an integer scalar indicating the integer variables in the model}
#' \item{n_fac_vars}{an integer scalar indicating the factor variables in the model}
#' \item{tree_leaves}{a list containing all the leaves in the forest}
#' \item{yc}{a list containing the target variable encoding scheme}
#' @export 
#' 
brif.default <- function(x, n_numeric_cuts = 31, n_integer_cuts = 31, max_integer_classes = 20, max_depth = 10, min_node_size = 1, ntrees = 100, ps = 0, max_factor_levels = 30, seed = 0, bagging_method = 0, bagging_proportion = 0.9, split_search = 3, search_radius = 5, verbose = 0, nthreads = 2, ...){

  # check argument validity
  if(ntrees < 1){
    stop("ntrees must be a positive integer.")
  }
  if(!split_search %in% c(0,1,2,3)){
    stop("Invalid value for split_search.")
  }
  if(search_radius < 1){
    stop("Search radius must be a positive integer.")
  }
  if(!bagging_method %in% c(0,1)){
    stop("Invalid value for bagging_method.")
  }
  if(ps < 0){
    stop("Invalid value for ps.")
  }
  if(n_numeric_cuts < 1){
    stop("n_numeric_cuts must be a positive integer.")
  }
  if(n_integer_cuts < 1){
    stop("n_integer_cuts must be a positive integer.")
  }
  if(max_integer_classes < 1){
    stop("max_integer_classes must be a positive integer.")
  }
  if(max_depth < 1){
    stop("max_depth must be a positive integer.")
  }
  if(min_node_size < 1){
    stop("min_node_size must be a positive integer.")
  }
  if(seed < 0){
    stop("seed must be a positive integer.")
  }
  
  n <- nrow(x)
  varnames <- colnames(x)
  vartypes <- sapply(x, class)
  # check for white spaces
  for(j in 1:length(vartypes)){
    if(vartypes[j] == "logical"){
      # change logical to integer
      x[,j] <- 1L*x[,j]
      vartypes[j] <- "integer"
      if(verbose) message("Casting logical variable ", varnames[j], " to integer.")
    }
    if(!vartypes[j] %in% c("factor","numeric","integer")){
      stop(paste(varnames[j],"is", vartypes[j], ". All variables must be a factor, numeric or integer type."))
    }
    if(vartypes[j] == 'factor'){
      # check for unique levels in factor
      n_uniques = length(unique(x[,j]))
      if(n_uniques > max_factor_levels) stop(paste("Variable", varnames[j], "has", n_uniques, "unique levels. If this is intended, adjust max_factor_levels parameter and re-run."))
    }
  }
  
  if(seed != 0) set.seed(as.integer(seed))
  # pad data and do stratified partition
  n_discard_bits <- ifelse(n %% 32 == 0, 0, 32 - n %% 32)
  if(n_discard_bits > 0){
    if(n < n_discard_bits) stop("Too few training data points. At least 16 is needed.")
    x_pad <- x[sample(1:n, n_discard_bits),]
    x <- rbind(x, x_pad)
    n <- nrow(x)
  }
  if(n == 32 & split_search >= 3){  # need at least two blocks
    x <- rbind(x,x)  # duplicate x
  }
  n <- nrow(x)
  
  x <- x[sample(1:n),]  # shuffle the rows
  x <- stratpar(x, 32)
  
  if(n < 128){
    bagging_proportion = 1
  }

  return(rftrain(x, par = list(n_numeric_cuts=as.integer(n_numeric_cuts), 
                              n_integer_cuts=as.integer(n_integer_cuts), 
                              max_integer_classes=as.integer(max_integer_classes), 
                              max_depth=as.integer(max_depth), 
                              min_node_size=as.integer(min_node_size), 
                              ntrees=as.integer(ntrees), 
                              ps=as.integer(ps), 
                              bagging_method=as.integer(bagging_method),
                              bagging_proportion=bagging_proportion,
                              split_search=as.integer(split_search),
                              search_radius=as.integer(search_radius),
                              verbose=as.integer(verbose), 
                              nthreads=as.integer(nthreads))))
}


#' Make predictions using a brif model
#' 
#' Make predictions for \code{newdata} using a brif model \code{object}. 
#' 
#' Note: If a model is built just for making predictions on one test set (i.e., no need to save the model object for future use), then the \code{\link[brif]{brif.trainpredict}} should be used. 
#' 
#' @param object an object of class "brif" as returned by the brif training function.
#' @param newdata a data frame. The predictor column names and data types must match those supplied for training. The order of the predictor columns does not matter though. 
#' @param type a character string indicating the return content. For a classification problem, "score" means the by-class probabilities and "class" means the class labels (i.e., the target variable levels). For regression, the predicted values are returned. #' @param vote_method an integer (0 or 1) specifying the voting method in prediction. 0: each leaf contributes the raw count; 1: each leaf contributes a fraction.
#' @param vote_method an integer (0 or 1) specifying the voting method in prediction. 0: each leaf contributes the raw count and an average is taken on the sum over all leaves; 1: each leaf contributes an intra-node fraction which is then averaged over all leaves with equal weight. 
#' @param nthreads an integer specifying the number of threads used by the program. Default is 2. This parameter only takes effect on Linux. On Mac OS or Windows, one thread will be used. 
#' @param ... additional arguments.
#' @return a data frame or a vector containing the prediction results. For regression, a numeric vector of predicted values will be returned. For classification, if \code{type = "class"}, a character vector of the predicted class labels will be returned; if \code{type = "score"}, a data frame will be returned, in which each column contains the probability of the new case being in the corresponding class. 
#' 
#' @examples 
#' # Predict using a model built by brif
#' pred_score <- predict(brif(Species ~ ., data = iris), iris, type = 'score')
#' pred_label <- predict(brif(Species ~ ., data = iris), iris, type = 'class')
#' 
#' # Equivalently and more efficiently:
#' pred_score <- brif(Species ~., data = iris, newdata = iris, type = 'score')
#' pred_label <- brif(Species ~., data = iris, newdata = iris, type = 'class')
#' 
#' # Or, retrieve predicted labels from the scores:
#' pred_label <- colnames(pred_score)[apply(pred_score, 1, which.max)]
#' 
#' 
#' @export
#' 
predict.brif <- function(object, newdata = NULL, type = c("score", "class"), vote_method = 1, nthreads = 2, ...) {
  if (!inherits(object, "brif")) stop("Not a legitimate \"brif\" object")
  type <- match.arg(type)
  if (is.null(newdata)) stop("newdata must be provided.")
  if(!is.data.frame(newdata)) stop("newdata is not a data frame.")
  if(!vote_method %in% c(0,1)) stop("Invalid value for vote_method.")
  if(nthreads < 0) stop("Invalid value for nthreads.")
  
  n <- nrow(newdata)
  if(n < 1) stop("newdata contains too few records.")
  # remove white spaces
  varnames <- object$var_labels
  vartypes <- object$var_types
  # check if all needed variables are present
  allcols <- colnames(newdata)
  for(name in varnames[2:length(varnames)]){
    if(!name %in% allcols) stop("The variable ", name, " is not present in newdata.")
  }
  for(j in 2:length(vartypes)){
    this_var_name = varnames[j]
    if(vartypes[j] == 'factor'){
      # do nothing    
    } else if(vartypes[j] == 'integer'){
      if(is.logical(newdata[,this_var_name])){
        newdata[,this_var_name] <- 1L*newdata[,this_var_name]  # cast logical to integer
      }
    }
  }
  pred <- rfpredict(object, newdata, as.integer(vote_method), as.integer(nthreads))
  if(type == 'score'){
    if(ncol(pred) <= 1){
      return(pred$pred)
    } else {
      # if the column names all start with "X", remove "X" from the names
      resp.labels <- colnames(pred)
      no_need_to_change <- 0
      for(i in 1:length(resp.labels)){
        if(substr(as.character(resp.labels[i]),1,1) != "X"){
          no_need_to_change <- 1
          break
        }
      }      
      if(no_need_to_change == 0){
        for(i in 1:length(resp.labels)){
          if(substr(as.character(resp.labels[i]),1,1) == "X"){
            resp.labels[i] <- sub('X','',resp.labels[i])
          }
        }   
        colnames(pred) <- resp.labels
      }
      return(pred)
    }
  } else if (type == 'class'){
    # if the column names all start with "X", remove "X" from the names
    resp.labels <- colnames(pred)
    no_need_to_change <- 0
    for(i in 1:length(resp.labels)){
      if(substr(as.character(resp.labels[i]),1,1) != "X"){
        no_need_to_change <- 1
        break
      }
    }      
    if(no_need_to_change == 0){
      for(i in 1:length(resp.labels)){
        if(substr(as.character(resp.labels[i]),1,1) == "X"){
          resp.labels[i] <- sub('X','',resp.labels[i])
        }
      }   
      colnames(pred) <- resp.labels
    }
    return(colnames(pred)[apply(pred, 1, which.max)])
  } else {
    return(pred)
  }
}

#' Train a model and use it to predict new cases
#' 
#' If the model is built to predict for just one test data set (newdata), then this function should be used instead of the \code{brif} and \code{predict.brif} pipeline. Transporting the model object between the training and prediction functions through saving and loading the \code{brif} object takes a subtantial amount of time, and using the \code{pred.trainpredict} function eliminates such time-consuming operations. This function will be automatically invoked by the \code{\link[brif]{brif}} function when the newdata argument is supplied there. 

#' @param x a data frame containing the training data set. The first column is taken as the target variable and all other columns are used as predictors. 
#' @param newdata a data frame containing the new data to be predicted. All columns in x (except for the first column which is the target variable) must be present in newdata and the data types must match. 
#' @param type a character string specifying the prediction format. Available values include "score" and "class". Default is "score". 
#' @param n_numeric_cuts an integer value indicating the maximum number of split points to generate for each numeric variable. 
#' @param n_integer_cuts an integer value indicating the maximum number of split points to generate for each integer variable. 
#' @param max_integer_classes an integer value. If the target variable is integer and has more than max_integer_classes unique values in the training data, then the target variable will be grouped into max_integer_classes bins. If the target variable is numeric, then the smaller of max_integer_classes and the number of unique values number of bins will be created on the target variables and the regression problem will be solved as a classification problem. 
#' @param max_depth an integer specifying the maximum depth of each tree. Default is 10. Maximum is 40. 
#' @param min_node_size an integer specifying the minimum number of training cases a leaf node must contain. Default is 1. 
#' @param ntrees an integer specifying the number of trees in the forest. 
#' @param seed an integer specifying the seed used by the internal random number generator. Default is 0, meaning not to set a seed but to accept the set seed from the calling environment.
#' @param ps an integer indicating the number of predictors to sample at each node split. Default is 0, meaning to use sqrt(p), where p is the number of predictors in the input. 
#' @param max_factor_levels an integer. If any factor variables has more than max_factor_levels, the program stops and prompts the user to increase the value of this parameter if the too-many-level factor is indeed intended.
#' @param bagging_method an integer indicating the bagging sampling method: 0 for sampling without replacement; 1 for sampling with replacement (bootstrapping). 
#' @param bagging_proportion a numeric scalar between 0 and 1, indicating the proportion of training observations to be used in each tree. 
#' @param split_search an integer indicating the choice of the split search method. 0: randomly pick a split point (fastest, greatest variance, least effective); 1: perform a greedy search starting from the mid-point; 2: evaluate every available split point and pick the best one; 3: perform a self-regulating local search to prevent over-fitting.
#' @param search_radius an positive integer indicating the split point search radius. This parameter takes effect only when split_search is 3. 
#' @param vote_method an integer (0 or 1) specifying the voting method in prediction. 0: each leaf contributes the raw count and an average is taken on the sum over all leaves; 1: each leaf contributes an intra-node fraction which is then averaged over all leaves with equal weight.  
#' @param verbose an integer (0 or 1) specifying the verbose level.  
#' @param nthreads an integer specifying the number of threads used by the program. Default is 2. This parameter takes effect only on systems supporting 'OpenMP'.  
#' @param ... additional arguments.
#' @return a data frame or a vector containing the prediction results. See \code{\link{predict.brif}} for details. 
#'
#' @examples 
#' trainset <- sample(1:nrow(iris), 0.5*nrow(iris))
#' validset <- setdiff(1:nrow(iris), trainset)
#' 
#' pred_score <- brif.trainpredict(iris[trainset, c(5,1:4)], iris[validset, c(1:4)], type = 'score')
#' pred_label <- colnames(pred_score)[apply(pred_score, 1, which.max)]
#'
#' @export 
#' 
brif.trainpredict <- function(x, newdata, type = c("score","class"), n_numeric_cuts = 31, n_integer_cuts = 31, max_integer_classes = 20, max_depth = 10, min_node_size = 1, ntrees=50, ps = 0, max_factor_levels = 30, seed = 0, bagging_method = 0, bagging_proportion = 0.9, vote_method = 1, split_search = 3, search_radius = 5, verbose = 0, nthreads = 2, ...){
  
  # check argument validity
  if(ntrees < 1){
    stop("ntrees must be a positive integer.")
  }
  if(!split_search %in% c(0,1,2,3)){
    stop("Invalid value for split_search.")
  }
  if(search_radius < 1){
    stop("Search radius must be a positive integer.")
  }
  if(!bagging_method %in% c(0,1)){
    stop("Invalid value for bagging_method.")
  }
  if(ps < 0){
    stop("Invalid value for ps.")
  }
  if(n_numeric_cuts < 1){
    stop("n_numeric_cuts must be a positive integer.")
  }
  if(n_integer_cuts < 1){
    stop("n_integer_cuts must be a positive integer.")
  }
  if(max_integer_classes < 1){
    stop("max_integer_classes must be a positive integer.")
  }
  if(max_depth < 1){
    stop("max_depth must be a positive integer.")
  }
  if(min_node_size < 1){
    stop("min_node_size must be a positive integer.")
  }
  if(seed < 0){
    stop("seed must be a positive integer.")
  }
  
  n <- nrow(x)
  varnames <- colnames(x)
  vartypes <- sapply(x, class)
  # check for white spaces
  for(j in 1:length(vartypes)){
    if(vartypes[j] == "logical"){
      # change logical to integer
      x[,j] <- 1L*x[,j] 
      vartypes[j] <- "integer"
      if(verbose) message("Casting logical variable ", varnames[j], " to integer.")
    }
    if(!vartypes[j] %in% c("factor","numeric","integer")){
      stop(paste(varnames[j],"is", vartypes[j], ". All variables must be a factor, numeric or integer type."))
    }
    if(vartypes[j] == 'factor'){
      # check for unique levels in factor
      n_uniques = length(unique(x[,j]))
      if(n_uniques > max_factor_levels) stop(paste("Variable", varnames[j], "has", n_uniques, "unique levels. If this is intended, adjust max_factor_levels parameter and re-run."))
    }
  }
  
  if(seed != 0) set.seed(seed)
  # pad data and do stratified partition
  n_discard_bits <- ifelse(n %% 32 == 0, 0, 32 - n %% 32)
  if(n_discard_bits > 0){
    if(n < n_discard_bits) stop("Too few training data points. At least 16 is needed.")
    x_pad <- x[sample(1:n, n_discard_bits),]
    x <- rbind(x, x_pad)
    n <- nrow(x)
  }
  if(n == 32 & split_search >= 3){  # need at least two blocks
    x <- rbind(x,x)  # duplicate x
    n <- nrow(x)
  }
  
  x <- x[sample(1:n),]  # shuffle the rows
  x <- stratpar(x, 32)
  
  if(n < 128){
    bagging_proportion = 1
  }
  
  # validity check for newdata
  if (is.null(newdata)) stop("newdata must be provided.")
  if(!is.data.frame(newdata)) stop("newdata is not a data frame.")
  
  n <- nrow(newdata)
  if(n < 1) stop("newdata contains too few records.")
  # remove white spaces
  # check if all needed variables are present
  allcols <- colnames(newdata)
  for(name in varnames[2:length(varnames)]){
    if(!name %in% allcols) stop("The variable ", name, " is not present in newdata.")
  }
  for(j in 2:length(vartypes)){
    this_var_name = varnames[j]
    if(vartypes[j] == 'factor'){
      # do nothing     
    } else if(vartypes[j] == 'integer'){
      if(is.logical(newdata[,this_var_name])){
        newdata[,this_var_name] <- 1L*newdata[,this_var_name]  # cast logical to integer
      }
    }
  }
  
  
  pred <- rftrainpredict(x, newdata, par = list(n_numeric_cuts=as.integer(n_numeric_cuts), 
                               n_integer_cuts=as.integer(n_integer_cuts), 
                               max_integer_classes=as.integer(max_integer_classes), 
                               max_depth=as.integer(max_depth), 
                               min_node_size=as.integer(min_node_size), 
                               ntrees=as.integer(ntrees), 
                               ps=as.integer(ps), 
                               bagging_method=as.integer(bagging_method),
                               bagging_proportion=bagging_proportion,
                               vote_method=as.integer(vote_method),
                               split_search=as.integer(split_search),
                               search_radius=as.integer(search_radius),
                               verbose=as.integer(verbose), 
                               nthreads=as.integer(nthreads)))
  this_type <- match.arg(type)
  if(this_type == 'score'){
    if(ncol(pred) <= 1){
      return(pred$pred)
    } else {
      # if the column names all start with "X", remove "X" from the names
      resp.labels <- colnames(pred)
      no_need_to_change <- 0
      for(i in 1:length(resp.labels)){
        if(substr(as.character(resp.labels[i]),1,1) != "X"){
          no_need_to_change <- 1
          break
        }
      }      
      if(no_need_to_change == 0){
        for(i in 1:length(resp.labels)){
          if(substr(as.character(resp.labels[i]),1,1) == "X"){
            resp.labels[i] <- sub('X','',resp.labels[i])
          }
        }   
        colnames(pred) <- resp.labels
      }
      return(pred)
    }
  } else if (this_type == 'class'){
    # if the column names all start with "X", remove "X" from the names
    resp.labels <- colnames(pred)
    no_need_to_change <- 0
    for(i in 1:length(resp.labels)){
      if(substr(as.character(resp.labels[i]),1,1) != "X"){
        no_need_to_change <- 1
        break
      }
    }      
    if(no_need_to_change == 0){
      for(i in 1:length(resp.labels)){
        if(substr(as.character(resp.labels[i]),1,1) == "X"){
          resp.labels[i] <- sub('X','',resp.labels[i])
        }
      }   
      colnames(pred) <- resp.labels
    }
    return(colnames(pred)[apply(pred, 1, which.max)])
  } else {
    return(pred)
  }
}


#' Build a single brif tree of a given depth
#' 
#' This is a wrapper for \code{\link[brif]{brif}} to build a single tree of a given depth. See \code{\link[brif]{brifTree.default}} and \code{\link[brif]{brifTree.formula}} for details. 
#' 
#' @param x a data frame or a \code{\link[stats]{formula}} object.
#' @param ... arguments passed on to \code{\link{brifTree.formula}} or \code{\link{brifTree.default}}. 
#' @return an object of class \code{brif}. See \code{\link{brif.default}} for details.
#' @examples 
#' # Build a single tree
#' bt <- brifTree(Species ~., data = iris, depth = 3)
#' 
#' # Print out the decision rules
#' printRules(bt)
#' 
#' # Get the accuracy on the training set
#' sum(predict(bt, newdata = iris, type = 'class') == iris[,'Species'])/nrow(iris)
#' 
#' @export
brifTree <- function(x, ...) UseMethod("brifTree");


#' Build a single brif tree taking a data frame as input
#' 
#' This function invokes \code{\link[brif]{brif.default}} with appropriately set parameters to generate a single tree with the maximum expected predictive accuracy.
#' 
#' @param x a data frame containing the training data. The first column is treated as the target variable. 
#' @param depth a positive integer indicating the desired depth of the tree. 
#' @param n_cuts a positive integer indicating the maximum number of split points to generate on each numeric or integer variable. A large value is preferred for a single tree. 
#' @param max_integer_classes a positive integer. See \code{\link[brif]{brif.default}} for details.
#' @param max_factor_levels a positive integer. See \code{\link[brif]{brif.default}} for details.
#' @param seed a non-negative positive integer specifying the random number generator seed. 
#' @param ... other relevant arguments.
#' @return an object of class \code{brif}. See \code{\link{brif.default}} for details.
#' 
#' @export
#' 
brifTree.default <- function(x, depth = 3, n_cuts = 2047, max_integer_classes = 20, max_factor_levels = 30, seed = 0, ...){

  return(brif.default(x, n_numeric_cuts = as.integer(n_cuts), n_integer_cuts = as.integer(n_cuts), max_integer_classes = as.integer(max_integer_classes), max_depth = depth, min_node_size = 1, ntrees = 1, ps = ncol(x), max_factor_levels = as.integer(max_factor_levels), seed = seed, bagging_method = 0, bagging_proportion = 1, split_search = 3, search_radius = as.integer(sqrt(n_cuts)),  verbose = 0, nthreads = 1, ...))
  
}


#' Build a single brif tree taking a formula as input
#'
#' @param formula an object of class "\code{\link[stats]{formula}}": a symbolic description of the model to be fitted.
#' @param data an optional data frame, list or environment (or object coercible by \code{\link{as.data.frame}} to a data frame) containing the variables in the model. If not found in \code{data}, the variables are taken from \code{environment(formula)}, typically the environment from which \code{brif.formula} is called.
#' @param subset an optional vector specifying a subset (in terms of index numbers, not actual data) of observations to be used in the fitting process.
#' @param na.action a function which indicates what should happen when the data contain NAs. 
#' @param depth a positive integer indicating the desired depth of the tree. 
#' @param n_cuts a positive integer indicating the maximum number of split points to generate on each numeric or integer variable. A large value is preferred for a single tree. 
#' @param max_integer_classes a positive integer. See \code{\link[brif]{brif.default}} for details.
#' @param max_factor_levels a positive integer. See \code{\link[brif]{brif.default}} for details.
#' @param seed a non-negative positive integer specifying the random number generator seed. 
#' @param ... other relevant arguments. 
#' @return an object of class \code{brif} to be used by \code{\link{predict.brif}}. 
#' 
#' @export
#' 
brifTree.formula <- function(formula, data, subset, na.action = stats::na.pass, depth = 3, n_cuts = 2047, max_integer_classes = 20, max_factor_levels = 30, seed = 0, ...) {
  Call <- match.call()
  indx <- match(c("formula", "data", "subset"), names(Call), nomatch = 0L)
  if(indx[1] == 0L) stop("a 'formula' argument is required")
  temp <- Call[c(1L, indx)]      # only keep the arguments we wanted
  temp$na.action <- na.action    # This one has a default
  temp[[1L]] <- quote(stats::model.frame) # change the function called
  mf <- eval.parent(temp)
  Terms <- attr(mf, "terms")
  return(brifTree.default(mf[,c(attr(mf, "names")[1],attr(Terms, "term.labels"))],
                          depth = depth, 
                          n_cuts = n_cuts,
                          max_integer_classes = max_integer_classes,
                          max_factor_levels = max_factor_levels,
                          seed = seed, ...))
}

#' Print the decision rules of a brif tree
#' 
#' @param object an object of class "brif" as returned by the brif training function.
#' @param which_tree a nonnegative integer indicating the tree number (starting from 0) in the forest to be printed. 
#' @return No return value. The function is called for side effect. The decision rules of the given tree is printed to the console output. Users can use \code{\link[base]{sink}} to direct the output to a file. 
#' 
#' @examples 
#' # Build a single tree
#' bt <- brifTree(Species ~., data = iris, depth = 3)
#' 
#' # Print out the decision rules
#' printRules(bt)
#' 
#' # Get the training accuracy
#' sum(predict(bt, newdata = iris, type = 'class') == iris[,'Species'])/nrow(iris)
#' 
#' @export
#' 
printRules <- function(object, which_tree = 0){
  if (!inherits(object, "brif")) stop("Not a legitimate \"brif\" object")
  if(which_tree < 0) stop("which_tree must be a nonnegative integer")
  printBrifTree(object, which_tree)
}
