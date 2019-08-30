
const Kron{T,N,I<:Tuple} = ApplyArray{T,N,typeof(kron),I}

function instantiate(K::Applied{<:Any,typeof(kron)})
    isempty(K.args) && throw(ArgumentError("Cannot take kronecker product of empty vectors"))
    applied(kron, map(instantiate,K.args)...)
end

Kron(A...) = ApplyArray(kron, A...)
Kron{T}(A...) where T = ApplyArray{T}(kron, A...)

_kron_dims() = 0
_kron_dims(A, B...) = max(ndims(A), _kron_dims(B...))

eltype(A::Applied{<:Any,typeof(kron)}) = promote_type(map(eltype,A.args)...)
ndims(A::Applied{<:Any,typeof(kron)}) = _kron_dims(A.args...)

size(K::Kron, j::Int) = prod(size.(K.args, j))
size(a::Kron{<:Any,1}) = (size(a,1),)
size(a::Kron{<:Any,2}) = (size(a,1), size(a,2))
size(a::Kron{<:Any,N}) where {N} = (@_inline_meta; ntuple(M -> size(a, M), Val(N)))
axes(a::Kron{<:Any,1}) = (OneTo(size(a,1)),)
axes(a::Kron{<:Any,2}) = (OneTo(size(a,1)), OneTo(size(a,2)))
axes(a::Kron{<:Any,N}) where {N} = (@_inline_meta; ntuple(M -> OneTo(size(a, M)), Val(N)))

getindex(K::Kron{T,1,<:Tuple{<:AbstractVector}}, k::Int) where T =
    first(K.args)[k]

function getindex(K::Kron{T,1,<:Tuple{<:AbstractVector,<:AbstractVector}}, k::Int) where T
    A,B = K.args
    K,κ = divrem(k-1, length(B))
    A[K+1]*B[κ+1]
end

getindex(K::Kron{T,2,<:Tuple{<:AbstractMatrix}}, k::Int, j::Int) where T =
    first(K.args)[k,j]

function getindex(K::Kron{T,2,<:Tuple{<:AbstractArray,<:AbstractArray}}, k::Int, j::Int) where T
    A,B = K.args
    K,κ = divrem(k-1, size(B,1))
    J,ξ = divrem(j-1, size(B,2))
    A[K+1,J+1]*B[κ+1,ξ+1]
end

## Adapted from julia/stdlib/LinearAlgebra/src/dense.jl kron definition
function _kron2!(R, K)
    size(R) == size(K) || throw(DimensionMismatch("Matrices have wrong dimensions"))
    a,b = K.args
    require_one_based_indexing(a, b)
    m = 1
    @inbounds for j = 1:size(a,2), l = 1:size(b,2), i = 1:size(a,1)
        aij = a[i,j]
        for k = 1:size(b,1)
            R[m] = aij*b[k,l]
            m += 1
        end
    end
    R
end

copyto!(R::AbstractMatrix, K::Kron{<:Any,2,<:Tuple{<:AbstractMatrix,<:AbstractMatrix}}) =
    _kron2!(R, K)
copyto!(R::AbstractVector, K::Kron{<:Any,1,<:Tuple{<:AbstractVector,<:AbstractVector}}) =
    _kron2!(R, K)


struct Diff{T, N, Arr} <: LazyArray{T, N}
    v::Arr
    dims::Int
end

Diff(v::AbstractVector{T}) where T = Diff{T,1,typeof(v)}(v, 1)

function Diff(A::AbstractMatrix{T}; dims::Integer) where T
    dims == 1 || dims == 2 || throw(ArgumentError("dimension must be 1 or 2, got $dims"))
    Diff{T,2,typeof(A)}(A, dims)
end

IndexStyle(::Type{<:Diff{<:Any,1}}) = IndexLinear()
IndexStyle(::Type{<:Diff{<:Any,2}}) = IndexCartesian()

size(D::Diff{<:Any,1}) = (length(D.v)-1,)
function size(D::Diff{<:Any,2})
    if D.dims == 1
        (size(D.v,1)-1,size(D.v,2))
    else #dims == 2
        (size(D.v,1),size(D.v,2)-1)
    end
end

getindex(D::Diff{<:Any, 1}, k::Integer) = D.v[k+1] - D.v[k]
function getindex(D::Diff, k::Integer, j::Integer)
    if D.dims == 1
        D.v[k+1,j] - D.v[k,j]
    else # dims == 2
        D.v[k,j+1] - D.v[k,j]
    end
end

struct Cumsum{T, N, Arr} <: LazyArray{T, N}
    v::Arr
    dims::Int
end

Cumsum(v::AbstractVector{T}) where T = Cumsum{T,1,typeof(v)}(v, 1)

function Cumsum(A::AbstractMatrix{T}; dims::Integer) where T
    dims == 1 || dims == 2 || throw(ArgumentError("dimension must be 1 or 2, got $dims"))
    Cumsum{T,2,typeof(A)}(A, dims)
end

IndexStyle(::Type{<:Cumsum{<:Any,1}}) = IndexLinear()
IndexStyle(::Type{<:Cumsum{<:Any,2}}) = IndexCartesian()

size(Q::Cumsum) = size(Q.v)

getindex(Q::Cumsum{<:Any, 1}, k::Integer) = k == 1 ? Q.v[1] : Q.v[k] + Q[k-1]
function getindex(Q::Cumsum, k::Integer, j::Integer)
    if Q.dims == 1
        k == 1 ? Q.v[1,j] : Q.v[k,j] + Q[k-1,j]
    else # dims == 2
        j == 1 ? Q.v[k,1] : Q.v[k,j] + Q[k,j-1]
    end
end

copyto!(x::AbstractArray{<:Any,N}, C::Cumsum{<:Any,N}) where N = cumsum!(x, C.v)

# keep lazy
cumsum(a::LazyArray; kwds...) = Cumsum(a; kwds...)