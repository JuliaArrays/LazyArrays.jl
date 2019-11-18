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

getindex(A::CachedVector, ::Colon) = copy(A)
getindex(A::CachedVector, ::Slice) = copy(A)

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

function resizedata!(B::CachedVector{T,Vector{T}}, n::Integer) where T<:Number
    @boundscheck checkbounds(Bool, B, n) || throw(ArgumentError("Cannot resize beyound size of operator"))

    # increase size of array if necessary
    olddata = B.data
    ν, = B.datasize
    n = max(ν,n)
    if n > length(B.data) # double memory to avoid O(n^2) growing
        B.data = Array{T}(undef, min(2n,length(B.array)))
        B.data[axes(olddata,1)] = olddata
    end

    inds = ν+1:n
    B.data[inds] .= view(B.array,inds)

    B.datasize = (n,)

    B
end

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


function convexunion(a::AbstractVector, b::AbstractVector) 
    isempty(a) && return b
    isempty(b) && return a
    min(minimum(a),minimum(b)):max(maximum(a),maximum(b))
end

colsupport(A::CachedMatrix, i) = 
    minimum(i) ≤ A.datasize[2] ? convexunion(colsupport(A.array, i),colsupport(A.data,i) ∩ Base.OneTo(A.datasize[1])) : colsupport(A.array, i)
colsupport(A::CachedVector, i) = 
    convexunion(colsupport(A.array, i),colsupport(A.data,i) ∩ Base.OneTo(A.datasize[1]))
rowsupport(A::CachedMatrix, i) = 
    minimum(i) ≤ A.datasize[1] ? convexunion(rowsupport(A.array, i),rowsupport(A.data,i) ∩ Base.OneTo(A.datasize[2])) : rowsupport(A.array, i)

replace_in_print_matrix(A::CachedMatrix, i::Integer, j::Integer, s::AbstractString) =
    i in colsupport(A,j) ? s : replace_with_centered_mark(s)


###
# special for zero cache
###

zero!(A::CachedArray{<:Any,N,<:Any,<:Zeros}) where N = zero!(A.data)
function getindex(A::CachedVector{T,<:AbstractVector,<:AbstractFill{<:Any,1}}, I::AbstractVector) where T
    @boundscheck checkbounds(A, I)
    CachedArray(A.data[I ∩ OneTo(A.datasize[1])], A.array[OneTo(length(I))])
end

###
# MemoryLayout
####

cachedlayout(_, _) = UnknownLayout()
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


function broadcasted(::LazyArrayStyle, op, A::CachedVector, B::AbstractVector)
    dat = paddeddata(A)
    n = length(dat)
    m = length(B)
    CachedArray(broadcast(op, dat, view(B,1:n)), broadcast(op, A.array, B))
end

function broadcasted(::LazyArrayStyle, op, A::AbstractVector, B::CachedVector)
    dat = paddeddata(B)
    n = length(dat)
    m = length(A)
    CachedArray(broadcast(op, view(A,1:n), dat), broadcast(op, A, B.array))
end

function broadcasted(::LazyArrayStyle, op, A::CachedVector, B::CachedVector)
    n = max(A.datasize[1],B.datasize[1])
    resizedata!(A,n)
    resizedata!(B,n)
    Adat = view(paddeddata(A),1:n)
    Bdat = view(paddeddata(B),1:n)
    CachedArray(broadcast(op, Adat, Bdat), broadcast(op, A.array, B.array))
end

###
# norm
###

norm1(a::CachedVector) = norm(paddeddata(a),1) + norm(@view(a.array[a.datasize[1]+1:end]),1)
norm2(a::CachedVector) = sqrt(norm(paddeddata(a),2)^2 + norm(@view(a.array[a.datasize[1]+1:end]),2)^2)
normInf(a::CachedVector) = max(norm(paddeddata(a),Inf), norm(@view(a.array[a.datasize[1]+1:end]),Inf))
normp(a::CachedVector, p) = (norm(paddeddata(a),2)^p + norm(@view(a.array[a.datasize[1]+1:end]),2)^p)^inv(p)
