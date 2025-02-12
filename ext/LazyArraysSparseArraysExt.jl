module LazyArraysSparseArraysExt

using SparseArrays: issparse, nnz, AbstractSparseArray
import LazyArrays: my_issparse, my_nnz

my_nnz(A::AbstractSparseArray) = nnz(A)

my_issparse(A::AbstractArray) = issparse(A)
my_issparse(A::DenseArray) = issparse(A)
my_issparse(S::AbstractSparseArray) = issparse(S)


end
