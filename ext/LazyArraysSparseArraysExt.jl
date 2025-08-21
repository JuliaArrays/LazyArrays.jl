module LazyArraysSparseArraysExt

using SparseArrays: issparse, nnz, AbstractSparseArray
import LazyArrays: local_issparse, local_nnz

local_nnz(A::AbstractSparseArray) = nnz(A)
local_issparse(A::AbstractArray) = issparse(A)

end
