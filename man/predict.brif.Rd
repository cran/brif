% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/brif.R
\name{predict.brif}
\alias{predict.brif}
\title{Make predictions using a brif model}
\usage{
\method{predict}{brif}(
  object,
  newdata = NULL,
  type = c("score", "class"),
  vote_method = 1,
  nthreads = 2,
  ...
)
}
\arguments{
\item{object}{an object of class "brif" as returned by the brif training function.}

\item{newdata}{a data frame. The predictor column names and data types must match those supplied for training. The order of the predictor columns does not matter though.}

\item{type}{a character string indicating the return content. For a classification problem, "score" means the by-class probabilities and "class" means the class labels (i.e., the target variable levels). For regression, the predicted values are returned.}

\item{vote_method}{an integer (0 or 1) specifying the voting method in prediction. 0: each leaf contributes the raw count and an average is taken on the sum over all leaves; 1: each leaf contributes an intra-node fraction which is then averaged over all leaves with equal weight.}

\item{nthreads}{an integer specifying the number of threads used by the program. This parameter only takes effect on systems supporting OpenMP.}

\item{...}{additional arguments.}
}
\value{
a data frame or a vector containing the prediction results. For regression, a numeric vector of predicted values will be returned. For classification, if \code{type = "class"}, a character vector of the predicted class labels will be returned; if \code{type = "score"}, a data frame will be returned, in which each column contains the probability of the new case being in the corresponding class.
}
\description{
Make predictions for \code{newdata} using a brif model \code{object}.
}
\details{
Note: If a model is built just for making predictions on one test set (i.e., no need to save the model object for future use), then the \code{\link[brif]{brif.trainpredict}} should be used.
}
\examples{
# Predict using a model built by brif
pred_score <- predict(brif(Species ~ ., data = iris), iris, type = 'score')
pred_label <- predict(brif(Species ~ ., data = iris), iris, type = 'class')

# Equivalently and more efficiently:
pred_score <- brif(Species ~., data = iris, newdata = iris, type = 'score')
pred_label <- brif(Species ~., data = iris, newdata = iris, type = 'class')

# Or, retrieve predicted labels from the scores:
pred_label <- colnames(pred_score)[apply(pred_score, 1, which.max)]


}
