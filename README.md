# Extending-the-OpenImageR-package-with-Gabor-feature-extraction

The details of the codeset and plots are included in the attached Microsoft Word Document (.docx) file in this repository. 
You need to view the file in "Read Mode" to see the contents properly after downloading the same.

The irlba and kernelKnn packages source code is attached with this repository as .ZIP file.

Irlba Package - A Brief Introduction
=====================================

Implicitly-restarted Lanczos methods for fast truncated singular value decomposition of sparse and dense matrices (also referred to as partial SVD). IRLBA stands for Augmented, Implicitly Restarted Lanczos Bidiagonalization Algorithm. The package provides the following functions (see help on each for details and examples).

        irlba() partial SVD function
        ssvd() l1-penalized matrix decompoisition for sparse PCA 
        prcomp_irlba() principal components function similar to the prcomp function in stats package for computing the first few principal components of large matrices
        svdr() alternate partial SVD function based on randomized SVD (see also the rsvd package by N. Benjamin Erichson for an alternative implementation)
        partial_eigen() a very limited partial eigenvalue decomposition for symmetric matrices (see the RSpectra package for more comprehensive truncated eigenvalue decomposition)

    library(irlba)
    set.seed(1)
    A <- matrix(rnorm(100), 10)

# ------------------ old way ----------------------------------------------
# A custom matrix multiplication function that scales the columns of A
# (cf the scale option). This function scales the columns of A to unit norm.
    col_scale <- sqrt(apply(A, 2, crossprod))
    mult <- function(x, y)
            {
              # check if x is a  vector
              if (is.vector(x))
              {
                return((x %*% y) / col_scale)
              }
              # else x is the matrix
              x %*% (y / col_scale)
            }
    irlba(A, 3, mult=mult)$d
## [1] 1.820227 1.622988 1.067185

# Compare with:
    irlba(A, 3, scale=col_scale)$d
                [1] 1.820227 1.622988 1.067185

# Compare with:
    svd(sweep(A, 2, col_scale, FUN=`/`))$d[1:3]
                [1] 1.820227 1.622988 1.067185

# ------------------ new way ----------------------------------------------
    setClass("scaled_matrix", contains="matrix", slots=c(scale="numeric"))
    setMethod("%*%", signature(x="scaled_matrix", y="numeric"), function(x ,y) x@.Data %*% (y / x@scale))
    setMethod("%*%", signature(x="numeric", y="scaled_matrix"), function(x ,y) (x %*% y@.Data) / y@scale)
    a <- new("scaled_matrix", A, scale=col_scale)

    irlba(a, 3)$d
                [1] 1.820227 1.622988 1.067185

We have learned that using R's existing S4 system is simpler, easier, and more flexible than using custom arguments with idiosyncratic syntax and behavior. We've even used the new approach to implement distributed parallel matrix products for very large problems with amazingly little code.
Wishlist / help wanted...    
