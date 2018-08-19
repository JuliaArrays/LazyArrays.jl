

function _Kron end

struct Kron{T,N,I} <: AbstractArray{T,N}
    arrays::I
    global function _Kron(A::I) where I<:Tuple{Vararg{<:AbstractArray{T,N}}} where {T,N}
        isempty(A) && throw(ArgumentError("Cannot take kronecker product of empty vectors"))
        new{T,N,I}(A)
    end
end

Kron{T,N}(A::AbstractArray{T,N}...) where {T,N} = _Kron(A)
Kron{T}(A::AbstractArray{T,N}...) where {T,N} = Kron{T,N}(A...)
Kron{T}(A::AbstractArray{<:Any}...) where T = Kron{T}(convert.(AbstractArray{T}, A)...)
Kron(A...) = Kron{mapreduce(eltype, promote_type, A)}(A...)



size(K::Kron, j::Int) = prod(size.(K.arrays, j))
size(a::Kron{<:Any,1}) = (size(a,1),)
size(a::Kron{<:Any,2}) = (size(a,1), size(a,2))
size(a::Kron{<:Any,N}) where {N} = (@_inline_meta; ntuple(M -> size(a, M), Val(N)))

getindex(K::Kron{T,1,<:Tuple{<:AbstractVector}}, k::Int) where T =
    first(K.arrays)[k]

function getindex(K::Kron{T,1,<:NTuple{2,<:AbstractVector}}, k::Int) where T
    A,B = K.arrays
    K,κ = divrem(k-1, length(B))
    A[K+1]*B[κ+1]
end

getindex(K::Kron{T,2,<:Tuple{<:AbstractMatrix}}, k::Int, j::Int) where T =
    first(K.arrays)[k,j]

function getindex(K::Kron{T,2,<:NTuple{2,<:AbstractMatrix}}, k::Int, j::Int) where T
    A,B = K.arrays
    K,κ = divrem(k-1, size(B,1))
    J,ξ = divrem(j-1, size(B,2))
    A[K+1,J+1]*B[κ+1,ξ+1]
end

## Adapted from julia/stdlib/LinearAlgebra/src/dense.jl kron definition
function _kron2!(R, K)
    a,b = K.arrays
    @assert !has_offset_axes(a, b)
    m = 1
    for j = 1:size(a,2), l = 1:size(b,2), i = 1:size(a,1)
        aij = a[i,j]
        for k = 1:size(b,1)
            R[m] = aij*b[k,l]
            m += 1
        end
    end
    R
end

copyto!(R::AbstractMatrix, K::Kron{<:Any,2,<:NTuple{2,<:AbstractMatrix}}) =
    _kron2!(R, K)
copyto!(R::AbstractVector, K::Kron{<:Any,1,<:NTuple{2,<:AbstractVector}}) =
    _kron2!(R, K)
