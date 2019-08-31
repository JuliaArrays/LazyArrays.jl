## CachedOperator

mutable struct CachedArray{T,N,DM<:AbstractArray{T,N},M<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::DM
    array::M
    datasize::NTuple{N,Int}
end

const CachedVector{T,DM<:AbstractVector{T},M<:AbstractVector{T}} = CachedArray{T,1,DM,M}
const CachedMatrix{T,DM<:AbstractMatrix{T},M<:AbstractMatrix{T}} = CachedArray{T,2,DM,M}

# CachedArray(data::AbstractArray{T,N}, array::AbstractArray{T,N}, sz::NTuple{N,Int}) where {T,N} =
#     CachedArray{T,N,typeof(data),typeof(array)}(data, array, sz)
CachedArray(data::AbstractArray, array::AbstractArray) = CachedArray(data, array, size(data))

CachedArray(::Type{Array}, array::AbstractArray{T,N}) where {T,N} =
    CachedArray(Array{T,N}(undef, ntuple(zero,N)), array)


CachedArray(array::AbstractArray{T,N}) where {T,N} =
    CachedArray(similar(array, ntuple(zero,N)), array)

"""
    cache(array::AbstractArray)

Caches the entries of an array.
"""
cache(::Type{MT}, O::AbstractArray) where {MT<:AbstractArray} = CachedArray(MT,O;kwds...)
cache(A::AbstractArray) = _cache(MemoryLayout(typeof(A)), A)
_cache(_, O::AbstractArray) = CachedArray(O)
_cache(_, O::CachedArray) = CachedArray(copy(O.data), O.array, O.datasize)
_cache(::AbstractStridedLayout, O::AbstractArray) = copy(O)



convert(::Type{AbstractArray{T}}, S::CachedArray{T}) where {T} = S
convert(::Type{AbstractArray{T}}, S::CachedArray) where {T} =
    CachedArray(convert(AbstractArray{T}, S.data), convert(AbstractArray{T}, S.array), S.datasize)


size(A::CachedArray) = size(A.array)
length(A::CachedArray) = length(A.array)

@propagate_inbounds function Base.getindex(B::CachedArray{T,N}, kj::Vararg{Integer,N}) where {T,N}
    @boundscheck checkbounds(B, kj...)
    resizedata!(B, kj...)
    B.data[kj...]
end

@propagate_inbounds function Base.getindex(B::CachedArray{T,1}, k::Integer) where T
    @boundscheck checkbounds(B, k)
    resizedata!(B, k)
    B.data[k]
end

@propagate_inbounds function Base.setindex!(B::CachedArray{T,N}, v, kj::Vararg{Integer,N}) where {T,N}
    @boundscheck checkbounds(B, kj...)
    resizedata!(B,kj...)
    @inbounds B.data[kj...] = v
    v
end

_maximum(ax, I) = maximum(I)
_maximum(ax, ::Colon) = maximum(ax)
function getindex(A::CachedArray, I...)
    @boundscheck checkbounds(A, I...)
    resizedata!(A, _maximum.(axes(A), I)...)
    A.data[I...]
end

function getindex(A::CachedVector, I, J...)
    @boundscheck checkbounds(A, I, J...)
    resizedata!(A, _maximum(axes(A,1), I))
    A.data[I]
end

function getindex(A::CachedVector, I::CartesianIndex)
    resizedata!(A, Tuple(I)...)
    A.data[I]
end

function getindex(A::CachedArray, I::CartesianIndex)
    resizedata!(A, Tuple(I)...)
    A.data[I]
end


## Array caching

function resizedata!(B::CachedArray{T,N,Array{T,N}},nm::Vararg{Integer,N}) where {T<:Number,N}
    @boundscheck checkbounds(Bool, B, nm...) || throw(ArgumentError("Cannot resize beyound size of operator"))

    # increase size of array if necessary
    olddata = B.data
    νμ = size(olddata)
    nm = max.(νμ,nm)
    if νμ ≠ nm
        B.data = Array{T}(undef, nm)
        B.data[axes(olddata)...] = olddata
    end

    for k in 1:N-1
        inds = tuple(axes(B.data)[1:k-1]...,νμ[k]+1:nm[k],Base.OneTo.(B.datasize[k+1:end])...)
        B.data[inds...] .= view(B.array,inds...)
    end
    let k = N
        inds = tuple(axes(B.data)[1:k-1]...,νμ[k]+1:nm[k])
        B.data[inds...] .= view(B.array,inds...)
    end
    B.datasize = nm

    B
end


_minimum(a) = isempty(a) ? length(a)+1 : minimum(a)
_maximum(a) = isempty(a) ? 0 : maximum(a)
convexunion(a::AbstractVector, b::AbstractVector) = min(_minimum(a),_minimum(b)):max(_maximum(a),_maximum(b))

colsupport(A::CachedMatrix, i) = convexunion(colsupport(A.array, i),colsupport(A.data,i))
colsupport(A::CachedVector, i) = convexunion(colsupport(A.array, i),colsupport(A.data,i))
rowsupport(A::CachedMatrix, i) = convexunion(rowsupport(A.array, i),rowsupport(A.data,i))

Base.replace_in_print_matrix(A::CachedMatrix, i::Integer, j::Integer, s::AbstractString) =
    i in colsupport(A,j) ? s : Base.replace_with_centered_mark(s)


###
# special for zero cache
###

zero!(A::CachedVector{<:Any,<:Any,<:Zeros}) = zero!(A.data)