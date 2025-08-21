module LazyArraysSparseArraysExt

using SparseArrays: issparse, nnz, AbstractSparseArray
import LazyArrays: local_issparse, local_nnz

#
# Add methods to local_nnz and local_issparse calling back to the corresponding
# methods defined in SparseArrays in case SparseArrays is loaded.
# 
local_nnz(A::AbstractSparseArray) = nnz(A)
local_issparse(A::AbstractArray) = issparse(A)

end
