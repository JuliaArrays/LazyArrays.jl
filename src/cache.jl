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
cache(A::AbstractArray) = _cache(MemoryLayout(A), A)
_cache(_, O::AbstractArray) = CachedArray(O)
_cache(_, O::CachedArray) = CachedArray(copy(O.data), O.array, O.datasize)
_cache(::AbstractStridedLayout, O::AbstractArray) = copy(O)



convert(::Type{AbstractArray{T}}, S::CachedArray{T}) where {T} = S
convert(::Type{AbstractArray{T}}, S::CachedArray) where {T} =
    CachedArray(convert(AbstractArray{T}, S.data), convert(AbstractArray{T}, S.array), S.datasize)


size(A::CachedArray) = size(A.array)
length(A::CachedArray) = length(A.array)

@propagate_inbounds function Base.getindex(B::CachedArray{T,N}, kj::Vararg{Integer,N}) where {T,N}
    @boundscheck checkbounds(Bool, B, kj...)
    resizedata!(B, kj...)
    B.data[kj...]
end

@propagate_inbounds function Base.setindex!(B::CachedArray{T,N}, v, kj::Vararg{Integer,N}) where {T,N}
    @boundscheck checkbounds(Bool, B, kj...)
    resizedata!(B,kj...)
    @inbounds B.data[kj...] = v
    v
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
