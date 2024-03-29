% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/brif.R
\name{brif.trainpredict}
\alias{brif.trainpredict}
\title{Train a model and use it to predict new cases}
\usage{
\method{brif}{trainpredict}(
  x,
  newdata,
  type = c("score", "class"),
  n_numeric_cuts = 31,
  n_integer_cuts = 31,
  max_integer_classes = 20,
  max_depth = 20,
  min_node_size = 1,
  ntrees = 200,
  ps = 0,
  max_factor_levels = 30,
  seed = 0,
  bagging_method = 0,
  bagging_proportion = 0.9,
  vote_method = 1,
  split_search = 4,
  search_radius = 5,
  verbose = 0,
  nthreads = 2,
  CUDA = 0,
  CUDA_blocksize = 128,
  CUDA_n_lb_GPU = 20480,
  cubrif_main = "cubrif_main.exe",
  tmp_file_prefix = "cbf",
  ...
)
}
\arguments{
\item{x}{a data frame containing the training data set. The first column is taken as the target variable and all other columns are used as predictors.}

\item{newdata}{a data frame containing the new data to be predicted. All columns in x (except for the first column which is the target variable) must be present in newdata and the data types must match.}

\item{type}{a character string specifying the prediction format. Available values include "score" and "class". Default is "score".}

\item{n_numeric_cuts}{an integer value indicating the maximum number of split points to generate for each numeric variable.}

\item{n_integer_cuts}{an integer value indicating the maximum number of split points to generate for each integer variable.}

\item{max_integer_classes}{an integer value. If the target variable is integer and has more than max_integer_classes unique values in the training data, then the target variable will be grouped into max_integer_classes bins. If the target variable is numeric, then the smaller of max_integer_classes and the number of unique values number of bins will be created on the target variables and the regression problem will be solved as a classification problem.}

\item{max_depth}{an integer specifying the maximum depth of each tree. Maximum is 40.}

\item{min_node_size}{an integer specifying the minimum number of training cases a leaf node must contain.}

\item{ntrees}{an integer specifying the number of trees in the forest.}

\item{ps}{an integer indicating the number of predictors to sample at each node split. Default is 0, meaning to use sqrt(p), where p is the number of predictors in the input.}

\item{max_factor_levels}{an integer. If any factor variables has more than max_factor_levels, the program stops and prompts the user to increase the value of this parameter if the too-many-level factor is indeed intended.}

\item{seed}{an integer specifying the seed used by the internal random number generator. Default is 0, meaning not to set a seed but to accept the set seed from the calling environment.}

\item{bagging_method}{an integer indicating the bagging sampling method: 0 for sampling without replacement; 1 for sampling with replacement (bootstrapping).}

\item{bagging_proportion}{a numeric scalar between 0 and 1, indicating the proportion of training observations to be used in each tree.}

\item{vote_method}{an integer (0 or 1) specifying the voting method in prediction. 0: each leaf contributes the raw count and an average is taken on the sum over all leaves; 1: each leaf contributes an intra-node fraction which is then averaged over all leaves with equal weight.}

\item{split_search}{an integer indicating the choice of the split search method. 0: randomly pick a split point; 1: do a local search; 2: random pick subject to regulation; 3: local search subject to regulation; 4 or above: a mix of options 0 to 3.}

\item{search_radius}{an positive integer indicating the split point search radius. This parameter takes effect only in regulated search (split_search = 2 or above).}

\item{verbose}{an integer (0 or 1) specifying the verbose level.}

\item{nthreads}{an integer specifying the number of threads used by the program. This parameter takes effect only on systems supporting OpenMP.}

\item{CUDA}{an integer (0, 1 or 2). 0: Do not use GPU. 1: Use GPU to build the forest. 2: Hybrid mode: Use GPU to split a node only when the node size is greater than CUDA_n_lb_GPU.}

\item{CUDA_blocksize}{a positive integer specifying the CUDA thread block size, must be a multiple of 64 up to 1024.}

\item{CUDA_n_lb_GPU}{a positive integer. The number of training cases must be greater than this number to enable the GPU computing when GPU = 2.}

\item{cubrif_main}{a string containing the path and name of the cubrif executable (see https://github.com/profyliu/cubrif for how to build it).}

\item{tmp_file_prefix}{a string for the path and prefix of temporary files created when CUDA is used.}

\item{...}{additional arguments.}
}
\value{
a data frame or a vector containing the prediction results. See \code{\link{predict.brif}} for details.
}
\description{
If the model is built to predict for just one test data set (newdata), then this function should be used instead of the \code{brif} and \code{predict.brif} pipeline. Transporting the model object between the training and prediction functions through saving and loading the \code{brif} object takes a subtantial amount of time, and using the \code{pred.trainpredict} function eliminates such time-consuming operations. This function will be automatically invoked by the \code{\link[brif]{brif}} function when the newdata argument is supplied there.
If GPU is used for training (GPU = 1 or 2), the total execution time of this function includes writing and reading temporary data files. To see timing of different steps, use verbose = 1. 
Note: Using GPU for training can improve training time only when the number of rows in the training data is extremely large, e.g., over 1 million. Even in such cases, GPU = 2 (hybrid mode) is recommended over GPU = 1 (force using GPU).
}
\examples{
trainset <- sample(1:nrow(iris), 0.5*nrow(iris))
validset <- setdiff(1:nrow(iris), trainset)

pred_score <- brif.trainpredict(iris[trainset, c(5,1:4)], iris[validset, c(1:4)], type = 'score')
pred_label <- colnames(pred_score)[apply(pred_score, 1, which.max)]

}
