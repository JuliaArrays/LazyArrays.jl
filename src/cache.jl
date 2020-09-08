## CachedOperator

mutable struct CachedArray{T,N,DM<:AbstractArray{T,N},M<:AbstractArray{T,N}} <: LazyArray{T,N}
    data::DM
    array::M
    datasize::NTuple{N,Int}
    function CachedArray{T,N,DM,M}(data::DM, array::M, datasize::NTuple{N,Int}) where {T,N,DM<:AbstractArray{T,N},M<:AbstractArray{T,N}}
        for d in datasize
            d < 0 && throw(ArgumentError("Datasize must be 0 or more"))
        end
        new{T,N,DM,M}(data, array, datasize)
    end
end

CachedArray(data::DM, array::M, datasize::NTuple{N,Int}) where {T,N,DM<:AbstractArray{T,N},M<:AbstractArray{T,N}} =
    CachedArray{T,N,DM,M}(data, array, datasize)

const CachedVector{T,DM<:AbstractVector{T},M<:AbstractVector{T}} = CachedArray{T,1,DM,M}
const CachedMatrix{T,DM<:AbstractMatrix{T},M<:AbstractMatrix{T}} = CachedArray{T,2,DM,M}

# CachedArray(data::AbstractArray{T,N}, array::AbstractArray{T,N}, sz::NTuple{N,Int}) where {T,N} =
#     CachedArray{T,N,typeof(data),typeof(array)}(data, array, sz)
CachedArray(data::AbstractArray, array::AbstractArray) = CachedArray(data, array, size(data))

# function CachedArray(::Type{Diagonal}, array::AbstractMatrix{T}) where T
#     axes(array,1) == axes(array,2) || throw(DimensionMismatch("Matrix must be square to cache as diagonal"))
#     CachedArray(Diagonal(Vector{T}(undef, size(array,1))), array)
# end

CachedArray(::Type{Array}, array::AbstractArray{T,N}) where {T,N} =
    CachedArray(Array{T,N}(undef, ntuple(zero,N)), array)


CachedArray(array::AbstractArray{T,N}) where {T,N} =
    CachedArray(similar(array, ntuple(zero,N)), array)

"""
    cache(array::AbstractArray)

Caches the entries of an array.
"""
cache(::Type{MT}, O::AbstractArray) where {MT<:AbstractArray} = CachedArray(MT,O)
cache(A::AbstractArray) = _cache(MemoryLayout(typeof(A)), A)
_cache(_, O::AbstractArray) = CachedArray(O)
_cache(_, O::CachedArray) = CachedArray(copy(O.data), O.array, O.datasize)
_cache(::AbstractStridedLayout, O::AbstractArray) = copy(O)



convert(::Type{AbstractArray{T}}, S::CachedArray{T}) where {T} = S
convert(::Type{AbstractArray{T}}, S::CachedArray) where {T} =
    CachedArray(convert(AbstractArray{T}, S.data), convert(AbstractArray{T}, S.array), S.datasize)

axes(A::CachedArray) = axes(A.array)
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

_maximum(ax, I::AbstractUnitRange{Int}) = maximum(I)
_maximum(ax, I) = maximum(ax[I])
_maximum(ax, ::Colon) = maximum(ax)
function getindex(A::CachedArray, I...)
    @boundscheck checkbounds(A, I...)
    resizedata!(A, _maximum.(axes(A), I)...)
    A.data[I...]
end

@inline getindex(A::CachedMatrix, kr::AbstractUnitRange, jr::AbstractUnitRange) = layout_getindex(A, kr, jr)
@inline getindex(A::CachedMatrix, kr::AbstractVector, jr::AbstractVector) = layout_getindex(A, kr, jr)
@inline getindex(A::CachedMatrix, ::Colon, j::Integer) = layout_getindex(A, :, j)

getindex(A::CachedVector, ::Colon) = copy(A)
getindex(A::CachedVector, ::Slice) = copy(A)

function cache_getindex(A::AbstractVector, I, J...)
    @boundscheck checkbounds(A, I, J...)
    resizedata!(A, _maximum(axes(A,1), I))
    A.data[I]
end

getindex(A::CachedVector, I, J...) = cache_getindex(A, I, J...)

function getindex(A::CachedVector, I::CartesianIndex)
    resizedata!(A, Tuple(I)...)
    A.data[I]
end

function getindex(A::CachedArray, I::CartesianIndex)
    resizedata!(A, Tuple(I)...)
    A.data[I]
end


## Array caching

resizedata!(B, mn...) = resizedata!(MemoryLayout(typeof(B.data)), MemoryLayout(typeof(B.array)), B, mn...)

function cache_filldata!(B, inds) 
    B.data[inds] .= view(B.array,inds)
end

function _vec_resizedata!(B::AbstractVector, n)
    @boundscheck checkbounds(Bool, B, n) || throw(ArgumentError("Cannot resize beyound size of operator"))

    # increase size of array if necessary
    olddata = paddeddata(B)
    ν, = B.datasize
    n = max(ν,n)
    if n > length(B.data) # double memory to avoid O(n^2) growing
        B.data = similar(B.data, min(2n,length(B)))
        B.data[axes(olddata,1)] = olddata
    end

    cache_filldata!(B, ν+1:n)
    B.datasize = (n,)

    B
end

resizedata!(_, _, B::AbstractVector, n) = _vec_resizedata!(B, n)
resizedata!(_, _, B::AbstractVector, n::Integer) = _vec_resizedata!(B, n)

function resizedata!(_, _, B::AbstractArray{<:Any,N}, nm::Vararg{Integer,N}) where N
    @boundscheck checkbounds(Bool, B, nm...) || throw(ArgumentError("Cannot resize beyound size of operator"))

    # increase size of array if necessary
    olddata = B.data
    νμ = size(olddata)
    nm = max.(νμ,nm)
    if νμ ≠ nm
        B.data = similar(B.data, nm...)
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

# sub array
function resizedata!(v::SubArray{<:Any,1,<:AbstractMatrix}, m::Integer)
    resizedata!(parent(v), m, parentindices(v)[2])
    v
end

convexunion(a::AbstractVector) = a

function convexunion(a::AbstractVector, b::AbstractVector)
    isempty(a) && return b
    isempty(b) && return a
    min(minimum(a),minimum(b)):max(maximum(a),maximum(b))
end

convexunion(a::AbstractVector, b::AbstractVector, c::AbstractVector...) =
    convexunion(convexunion(a,b), c...)

function colsupport(A::CachedMatrix, i)
    isempty(i) && return 1:0
    minimum(i) ≤ A.datasize[2] ? convexunion(colsupport(A.array, i),colsupport(A.data,i) ∩ Base.OneTo(A.datasize[1])) : colsupport(A.array, i)
end
colsupport(A::CachedVector, i) =
    convexunion(colsupport(A.array, i),colsupport(A.data,i) ∩ Base.OneTo(A.datasize[1]))
function rowsupport(A::CachedMatrix, i)
    isempty(i) && return 1:0
    minimum(i) ≤ A.datasize[1] ? convexunion(rowsupport(A.array, i),rowsupport(A.data,i) ∩ Base.OneTo(A.datasize[2])) : rowsupport(A.array, i)
end


###
# special for zero cache
###

function zero!(A::CachedArray{<:Any,N,<:Any,<:Zeros}) where N
    zero!(A.data)
    A
end
function _cached_getindex_vector(A, I)
    @boundscheck checkbounds(A, I)
    CachedArray(A.data[I ∩ OneTo(A.datasize[1])], A.array[OneTo(length(I))])
end

getindex(A::CachedVector{T,<:AbstractVector,<:AbstractFill{<:Any,1}}, I::AbstractVector) where T =
    _cached_getindex_vector(A, I)
getindex(A::CachedVector{T,<:AbstractVector,<:AbstractFill{<:Any,1}}, I::AbstractUnitRange) where T =
    _cached_getindex_vector(A, I)

###
# MemoryLayout
####

struct CachedLayout{Data,Array} <: MemoryLayout end

cachedlayout(::Data, ::Array) where {Data,Array} = CachedLayout{Data,Array}()
MemoryLayout(C::Type{CachedArray{T,N,DAT,ARR}}) where {T,N,DAT,ARR} = cachedlayout(MemoryLayout(DAT), MemoryLayout(ARR))



#####
# broadcasting
#
# We want broadcasting for numbers with concaenations to pass through
# to take advantage of special implementations of the sub-components
######

BroadcastStyle(::Type{<:CachedArray{<:Any,N}}) where N = LazyArrayStyle{N}()

broadcasted(::LazyArrayStyle, op, A::CachedArray) =
    CachedArray(broadcast(op, paddeddata(A)), broadcast(op, A.array))

broadcasted(::LazyArrayStyle, op, A::CachedArray, c::Number) =
    CachedArray(broadcast(op, paddeddata(A), c), broadcast(op, A.array, c))
broadcasted(::LazyArrayStyle, op, c::Number, A::CachedArray) =
CachedArray(broadcast(op, c, paddeddata(A)), broadcast(op, c, A.array))
broadcasted(::LazyArrayStyle, op, A::CachedArray, c::Ref) =
    CachedArray(broadcast(op, paddeddata(A), c), broadcast(op, A.array, c))
broadcasted(::LazyArrayStyle, op, c::Ref, A::CachedArray) =
    CachedArray(broadcast(op, c, paddeddata(A)), broadcast(op, c, A.array))


function _cache_broadcast(::CachedLayout, _, op, A, B)
    dat = paddeddata(A)
    n = length(dat)
    m = length(B)
    CachedArray(broadcast(op, dat, view(B,1:n)), broadcast(op, A.array, B))
end

function _cache_broadcast(_, ::CachedLayout, op, A, B)
    dat = paddeddata(B)
    n = length(dat)
    m = length(A)
    CachedArray(broadcast(op, view(A,1:n), dat), broadcast(op, A, B.array))
end

function _cache_broadcast(::CachedLayout, ::CachedLayout, op, A, B)
    n = max(A.datasize[1],B.datasize[1])
    resizedata!(A,n)
    resizedata!(B,n)
    Adat = view(paddeddata(A),1:n)
    Bdat = view(paddeddata(B),1:n)
    CachedArray(broadcast(op, Adat, Bdat), broadcast(op, A.array, B.array))
end

function cache_broadcast(op, A, B)
    if length(A) ≠ length(B)
        (length(A) == 1 || length(B) == 1) && error("Internal error: Scalar-like broadcasting not yet supported.")
        throw(DimensionMismatch("arrays could not be broadcast to a common size; got a dimension with lengths $(length(A)) and $(length(B))"))
    end
    _cache_broadcast(MemoryLayout(A), MemoryLayout(B), op, A, B)
end

broadcasted(::LazyArrayStyle, op, A::CachedVector, B::AbstractVector) = cache_broadcast(op, A, B)
broadcasted(::LazyArrayStyle, op, A::AbstractVector, B::CachedVector) = cache_broadcast(op, A, B)
broadcasted(::LazyArrayStyle, op, A::CachedVector, B::CachedVector) = cache_broadcast(op, A, B)

broadcasted(::LazyArrayStyle, op, A::SubArray{<:Any,1,<:CachedMatrix}, B::CachedVector) = cache_broadcast(op, A, B)
broadcasted(::LazyArrayStyle, op, A::SubArray{<:Any,1,<:CachedMatrix}, B::AbstractVector) = cache_broadcast(op, A, B)
broadcasted(::LazyArrayStyle, op, A::CachedVector, B::SubArray{<:Any,1,<:CachedMatrix}) = cache_broadcast(op, A, B)
broadcasted(::LazyArrayStyle, op, A::AbstractVector, B::SubArray{<:Any,1,<:CachedMatrix}) = cache_broadcast(op, A, B)



###
# norm
###

# allow overloading for special backends, e.g., padded
_norm2(_, a) = sqrt(norm(paddeddata(a),2)^2 + norm(@view(a.array[a.datasize[1]+1:end]),2)^2)
_norm1(_, a) = norm(paddeddata(a),1) + norm(@view(a.array[a.datasize[1]+1:end]),1)
_normInf(_, a) = max(norm(paddeddata(a),Inf), norm(@view(a.array[a.datasize[1]+1:end]),Inf))
_normp(_, a, p) = (norm(paddeddata(a),p)^p + norm(@view(a.array[a.datasize[1]+1:end]),p)^p)^inv(p)

norm1(a::CachedVector) = _norm1(MemoryLayout(a), a)
norm2(a::CachedVector) = _norm2(MemoryLayout(a), a)
normInf(a::CachedVector) = _normInf(MemoryLayout(a), a)
normp(a::CachedVector, p) = _normp(MemoryLayout(a), a, p)

###
# fill!/lmul!/rmul!
###

function fill!(a::CachedArray, x)
    fill!(a.data, x)
    fill!(a.array, x)
    a
end

function rmul!(a::CachedArray, x::Number)
    rmul!(a.data, x)
    rmul!(a.array, x)
    a
end

function lmul!(x::Number, a::CachedArray)
    lmul!(x, a.data)
    lmul!(x, a.array)
    a
end

lmul!(x::Number, a::SubArray{<:Any,N,<:CachedArray}) where N = ArrayLayouts.lmul!(x, a)
rmul!(a::SubArray{<:Any,N,<:CachedArray}, x::Number) where N = ArrayLayouts.rmul!(a, x)


###
# copy
###

# need to copy data to prevent mutation. `a.array` is never changed so does not need to be 
# copied
copy(a::CachedArray) = CachedArray(copy(a.data), a.array, a.datasize)
copy(a::Adjoint{<:Any,<:CachedArray}) = copy(parent(a))'
copy(a::Transpose{<:Any,<:CachedArray}) = transpose(copy(parent(a)))

