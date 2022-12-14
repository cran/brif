% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/brif.R
\name{printRules}
\alias{printRules}
\title{Print the decision rules of a brif tree}
\usage{
printRules(object, which_tree = 0)
}
\arguments{
\item{object}{an object of class "brif" as returned by the brif training function.}

\item{which_tree}{a nonnegative integer indicating the tree number (starting from 0) in the forest to be printed.}
}
\value{
No return value. The function is called for side effect. The decision rules of the given tree is printed to the console output. Users can use \code{\link[base]{sink}} to direct the output to a file.
}
\description{
Print the decision rules of a brif tree
}
\examples{
# Build a single tree
bt <- brifTree(Species ~., data = iris, depth = 3)

# Print out the decision rules
printRules(bt)

# Get the training accuracy
sum(predict(bt, newdata = iris, type = 'class') == iris[,'Species'])/nrow(iris)

}
