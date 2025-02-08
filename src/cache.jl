## CachedOperator

abstract type AbstractCachedArray{T,N} <: LazyArray{T,N} end
const AbstractCachedVector{T} = AbstractCachedArray{T,1}
const AbstractCachedMatrix{T} = AbstractCachedArray{T,2}


mutable struct CachedArray{T,N,DM<:AbstractArray{T,N},M} <: AbstractCachedArray{T,N}
    data::DM
    array::M
    datasize::NTuple{N,Int}
    function CachedArray{T,N,DM,M}(data::DM, array::M, datasize::NTuple{N,Int}) where {T,N,DM<:AbstractArray{T,N},M}
        for d in datasize
            d < 0 && throw(ArgumentError("Datasize must be 0 or more"))
        end
        new{T,N,DM,M}(data, array, datasize)
    end
end

CachedArray(data::AbstractArray{T,N}, array::AbstractArray{T,N}, datasize::NTuple{N,Int}) where {T,N} =
    CachedArray{T,N,typeof(data),typeof(array)}(data, array, datasize)

function CachedArray(data::AbstractArray{T,N}, array::AbstractArray{V,N}, datasize::NTuple{N,Int}) where {T,V,N}
    TV = promote_type(T,V)
    CachedArray(convert(AbstractArray{TV,N}, data), convert(AbstractArray{TV,N}, array), datasize)
end

const CachedVector{T,DM<:AbstractVector{T},M<:AbstractVector{T}} = CachedArray{T,1,DM,M}
const CachedMatrix{T,DM<:AbstractMatrix{T},M} = CachedArray{T,2,DM,M}



# CachedArray(data::AbstractArray{T,N}, array::AbstractArray{T,N}, sz::NTuple{N,Int}) where {T,N} =
#     CachedArray{T,N,typeof(data),typeof(array)}(data, array, sz)
CachedArray(data::AbstractArray, array::AbstractArray) = CachedArray(data, array, size(data))

# function CachedArray(::Type{Diagonal}, array::AbstractMatrix{T}) where T
#     axes(array,1) == axes(array,2) || throw(DimensionMismatch("Matrix must be square to cache as diagonal"))
#     CachedArray(Diagonal(Vector{T}(undef, size(array,1))), array)
# end

CachedArray(::Type{Array}, array::AbstractArray{T,N}) where {T,N} = CachedArray(Array{T,N}(undef, ntuple(zero,N)), array)
CachedArray(array::AbstractArray{T,N}) where {T,N} = CachedArray(similar(array, ntuple(zero,N)), array)

"""
    cache(array::AbstractArray)

Caches the entries of an array.
"""
cache(::Type{MT}, O::AbstractArray) where {MT<:AbstractArray} = CachedArray(MT,O)


cache(A::AbstractArray) = cache_layout(MemoryLayout(A), A)
cache_layout(_, O::AbstractArray) = CachedArray(O)
cache_layout(_, O::CachedArray) = CachedArray(copy(O.data), O.array, O.datasize)
cache_layout(::AbstractStridedLayout, O::AbstractArray) = copy(O)

const _cache = cache_layout # TODO: deprecate
cacheddata(A::AbstractCachedArray) = view(A.data,OneTo.(A.datasize)...)

convert(::Type{AbstractArray{T}}, S::CachedArray{T}) where T = S
convert(::Type{AbstractArray{T,N}}, S::CachedArray{T,N}) where {T,N} = S
convert(::Type{AbstractArray{T}}, S::CachedArray{<:Any,N}) where {T,N} = convert(AbstractArray{T,N}, S)
convert(::Type{AbstractArray{T,N}}, S::CachedArray{<:Any,N}) where {T,N} =
    CachedArray(convert(AbstractArray{T}, S.data), convert(AbstractArray{T}, S.array), S.datasize)

axes(A::CachedArray) = axes(A.array)
size(A::CachedArray) = size(A.array)
length(A::CachedArray) = length(A.array)

@propagate_inbounds function Base.getindex(B::AbstractCachedArray{T,N}, kj::Vararg{Integer,N}) where {T,N}
    @boundscheck checkbounds(B, kj...)
    resizedata!(B, kj...)
    B.data[kj...]
end

@propagate_inbounds function Base.getindex(B::AbstractCachedArray{T,1}, k::Integer) where T
    @boundscheck checkbounds(B, k)
    resizedata!(B, k)
    B.data[k]
end

@propagate_inbounds function Base.setindex!(B::AbstractCachedArray{T,N}, v, kj::Vararg{Integer,N}) where {T,N}
    @boundscheck checkbounds(B, kj...)
    resizedata!(B,kj...)
    @inbounds B.data[kj...] = v
    v
end

_maximum(ax, I::AbstractUnitRange{Int}) = maximum(I)
_maximum(ax, I) = maximum(ax[I])
_maximum(ax, ::Colon) = maximum(ax)
function getindex(A::AbstractCachedArray, I...)
    @boundscheck checkbounds(A, I...)
    resizedata!(A, _maximum.(axes(A), I)...)
    A.data[I...]
end

getindex(A::AbstractCachedMatrix, I::Integer) = A[Base._to_subscript_indices(A, I)...]

@inline function _isfinite_getindex(A::AbstractCachedMatrix, kr, jr, maxkr::Int, maxjr::Int)
    resizedata!(A, maxkr, maxjr)
    A.data[kr, jr]
end

@inline function _isfinite_getindex(A::AbstractCachedMatrix, kr, maxkr::Int)
    m,n = size(A)
    if maxkr < m
        resizedata!(A, maxkr, 1)
    else
        resizedata!(A, m, div(maxkr-1, m)+1)
    end
    A.data[kr]
end

# default is to do lazy
@inline _isfinite_getindex(A, kr, jr, _, _) = layout_getindex(A, kr, jr)
@inline _isfinite_getindex(A, kr, _) = layout_getindex(A, kr)

@inline getindex(A::AbstractCachedMatrix, k::Integer, jr::AbstractUnitRange) = _isfinite_getindex(A, k, jr, k, maximum(jr))
@inline getindex(A::AbstractCachedMatrix, kr::AbstractUnitRange, j::Integer) = _isfinite_getindex(A, kr, j, maximum(kr), j)
@inline getindex(A::AbstractCachedMatrix, kr::AbstractUnitRange, jr::AbstractUnitRange) = _isfinite_getindex(A, kr, jr, maximum(kr), maximum(jr))
@inline getindex(A::AbstractCachedMatrix, kr::AbstractVector, jr::AbstractVector) = _isfinite_getindex(A, kr, jr, maximum(kr), maximum(jr))
@inline getindex(A::AbstractCachedMatrix, k::Integer, jr::AbstractVector) = _isfinite_getindex(A, k, jr, k, maximum(jr))
@inline getindex(A::AbstractCachedMatrix, kr::AbstractVector, j::Integer) =  _isfinite_getindex(A, kr, j, maximum(kr), j)
@inline getindex(A::AbstractCachedMatrix, k::Integer, ::Colon) = _isfinite_getindex(A, k, :, k, size(A,2))
@inline getindex(A::AbstractCachedMatrix, kr::AbstractVector, ::Colon) = _isfinite_getindex(A, kr, :, maximum(kr), size(A,2))
@inline getindex(A::AbstractCachedMatrix, kr::AbstractUnitRange, ::Colon) = _isfinite_getindex(A, kr, :, maximum(kr), size(A,2))
@inline getindex(A::AbstractCachedMatrix, ::Colon, j::Integer) = _isfinite_getindex(A, :, j, size(A,1), j)
@inline getindex(A::AbstractCachedMatrix, ::Colon, jr::AbstractVector) = _isfinite_getindex(A, :, jr, size(A,1), maximum(jr))
@inline getindex(A::AbstractCachedMatrix, ::Colon, jr::AbstractUnitRange) = _isfinite_getindex(A, :, jr, size(A,1), maximum(jr))
@inline getindex(A::AbstractCachedMatrix, ::Colon, ::Colon) = _isfinite_getindex(A, :, :, size(A,1), size(A,2))
@inline getindex(A::AbstractCachedMatrix, kr::AbstractVector) = _isfinite_getindex(A, kr, maximum(kr))

# Structured Caching cases
for Tri in (:UnitUpperTriangular, :UpperTriangular, :UnitLowerTriangular, :LowerTriangular)
    @eval begin
        @inline function _isfinite_getindex(A::$Tri{T, <:AbstractCachedMatrix{T}}, kr, jr, maxkr::Int, maxjr::Int) where T
            nmax = max(maxkr,maxjr)
            resizedata!(A.data, nmax, nmax)
            getindex($Tri(A.data.data[1:nmax,1:nmax]), kr, jr)
        end
    end
end

for Adj in (:Adjoint, :Transpose)
    @eval begin
        @inline function _isfinite_getindex(A::$Adj{T, <:AbstractCachedMatrix{T}}, kr, jr, maxkr::Int, maxjr::Int) where T
            resizedata!(parent(A), maxjr, maxkr)
            $Adj(parent(A).data)[kr,jr]
        end

        @inline function _isfinite_getindex(A::$Adj{T, <:AbstractCachedMatrix{T}}, kr, maxkr::Int) where T
            m,n = size(A)
            if maxkr < m
                resizedata!(parent(A), 1, maxkr)
            else
                resizedata!(parent(A), div(maxkr-1, m)+1, m)
            end
            $Adj(parent(A).data)[kr]
        end

        @inline function _isfinite_getindex(A::$Adj{T, <:AbstractCachedVector{T}}, kr, jr, maxkr::Int, maxjr::Int) where T
            resizedata!(parent(A), maxjr)
            $Adj(parent(A).data)[kr,jr]
        end
    end
end

@inline _isfinite_getindex(A::Transpose{T, <:AbstractCachedVector{T}}, kr, maxkr::Int) where T = parent(A)[kr]
@inline _isfinite_getindex(A::Adjoint{T, <:AbstractCachedVector{T}}, kr, maxkr::Int) where T = conj(parent(A)[kr])


for Wrap in (:UpperOrLowerTriangular, :AdjOrTrans)
    @eval begin
        @inline getindex(A::$Wrap{T, <:AbstractCachedArray{T}}, k::Integer, jr::AbstractUnitRange) where T = _isfinite_getindex(A, k, jr, k, maximum(jr))
        @inline getindex(A::$Wrap{T, <:AbstractCachedArray{T}}, kr::AbstractUnitRange, j::Integer) where T = _isfinite_getindex(A, kr, j, maximum(kr), j)
        @inline getindex(A::$Wrap{T, <:AbstractCachedArray{T}}, kr::AbstractUnitRange, jr::AbstractUnitRange) where T = _isfinite_getindex(A, kr, jr, maximum(kr), maximum(jr))
        @inline getindex(A::$Wrap{T, <:AbstractCachedArray{T}}, kr::AbstractVector, jr::AbstractVector) where T = _isfinite_getindex(A, kr, jr, maximum(kr), maximum(jr))
        @inline getindex(A::$Wrap{T, <:AbstractCachedArray{T}}, k::Integer, jr::AbstractVector) where T = _isfinite_getindex(A, k, jr, k, maximum(jr))
        @inline getindex(A::$Wrap{T, <:AbstractCachedArray{T}}, kr::AbstractVector, j::Integer) where T =  _isfinite_getindex(A, kr, j, maximum(kr), j)
        @inline getindex(A::$Wrap{T, <:AbstractCachedArray{T}}, k::Integer, ::Colon) where T = _isfinite_getindex(A, k, :, k, size(A,2))
        @inline getindex(A::$Wrap{T, <:AbstractCachedArray{T}}, kr::AbstractVector, ::Colon) where T = _isfinite_getindex(A, kr, :, maximum(kr), size(A,2))
        @inline getindex(A::$Wrap{T, <:AbstractCachedArray{T}}, kr::AbstractUnitRange, ::Colon) where T = _isfinite_getindex(A, kr, :, maximum(kr), size(A,2))
        @inline getindex(A::$Wrap{T, <:AbstractCachedArray{T}}, ::Colon, j::Integer) where T = _isfinite_getindex(A, :, j, size(A,1), j)
        @inline getindex(A::$Wrap{T, <:AbstractCachedArray{T}}, ::Colon, jr::AbstractVector) where T = _isfinite_getindex(A, :, jr, size(A,1), maximum(jr))
        @inline getindex(A::$Wrap{T, <:AbstractCachedArray{T}}, ::Colon, jr::AbstractUnitRange) where T = _isfinite_getindex(A, :, jr, size(A,1), maximum(jr))
        @inline getindex(A::$Wrap{T, <:AbstractCachedArray{T}}, ::Colon, ::Colon) where T = _isfinite_getindex(A, :, :, size(A,1), size(A,2))
        @inline getindex(A::$Wrap{T, <:AbstractCachedArray{T}}, kr::AbstractVector) where T = _isfinite_getindex(A, kr, maximum(kr))
    end
end

getindex(A::AbstractCachedVector, ::Colon) = copy(A)
getindex(A::AbstractCachedVector, ::Slice) = copy(A)

function cache_getindex(_, A::AbstractVector, I, J...)
    n = _maximum(axes(A,1), I)
    @boundscheck checkbounds(A, I, J...)
    isempty(I) || resizedata!(A, n)
    A.data[I]
end
# allow dispatch on resize length for infinite arrays
cache_getindex(A::AbstractVector, I, J...) = cache_getindex(length(I), A, I, J...)

getindex(A::AbstractCachedVector, I, J...) = cache_getindex(A, I, J...)
getindex(A::AbstractCachedVector, I::AbstractVector) = cache_getindex(A, I)

function getindex(A::AbstractCachedVector, I::CartesianIndex)
    resizedata!(A, Tuple(I)...)
    A.data[I]
end

function getindex(A::AbstractCachedArray, I::CartesianIndex)
    resizedata!(A, Tuple(I)...)
    A.data[I]
end


## Array caching

resizedata!(B::CachedArray, mn...) = resizedata!(MemoryLayout(B.data), MemoryLayout(B.array), B, mn...)
resizedata!(B::AbstractCachedArray, mn...) = resizedata!(MemoryLayout(B.data), UnknownLayout(), B, mn...)
resizedata!(A, mn...) = A # don't do anything
function resizedata!(A::AdjOrTrans, m, n)
    m ≤ 0 || resizedata!(parent(A), n)
    A
end

function cache_filldata!(B, inds...)
    B.data[inds...] .= view(B.array,inds...)
end

function _vec_resizedata!(B::AbstractVector, n)
    n ≤ 0 && return B
    @boundscheck checkbounds(Bool, B, n) || throw(ArgumentError("Cannot resize beyond size of operator"))

    # increase size of array if necessary
    olddata = cacheddata(B)
    ν, = B.datasize
    n = max(ν,n)
    if n > length(B.data) # double memory to avoid O(n^2) growing
        B.data = similar(B.data, min(2n,length(B)))
        B.data[axes(olddata,1)] = olddata
    end
    if n > ν
        cache_filldata!(B, ν+1:n)
        B.datasize = (n,)
    end

    B
end

resizedata!(_, _, B::AbstractVector, n) = _vec_resizedata!(B, n)
resizedata!(_, _, B::AbstractVector, n::Integer) = _vec_resizedata!(B, n)

function resizedata!(_, _, B::AbstractArray{<:Any,N}, nm::Vararg{Integer,N}) where N
    @boundscheck all(nm .≤ size(B)) || throw(ArgumentError("Cannot resize beyond size of operator"))

    # increase size of array if necessary
    olddata = cacheddata(B)
    νμ = size(olddata)
    nm = max.(νμ,nm)
    if νμ ≠ nm
        B.data = similar(B.data, nm...)
        B.data[axes(olddata)...] = olddata
    end

    for k in 1:N-1
        inds = tuple(axes(B.data)[1:k-1]...,νμ[k]+1:nm[k],Base.OneTo.(B.datasize[k+1:end])...)
        cache_filldata!(B, inds...)
    end
    let k = N
        inds = tuple(axes(B.data)[1:k-1]...,νμ[k]+1:nm[k])
        cache_filldata!(B, inds...)
    end
    B.datasize = nm

    B
end


convexunion(a::AbstractVector) = a

function convexunion(a::AbstractVector, b::AbstractVector)
    isempty(a) && return b
    isempty(b) && return a
    min(minimum(a),minimum(b)):max(maximum(a),maximum(b))
end

convexunion(a::AbstractVector, b::AbstractVector, c::AbstractVector...) =
    convexunion(convexunion(a,b), c...)

convexunion(a::Number, b::Number, c...) = convexunion(a:a, b:b, c...)
convexunion(a::Number, b, c...) = convexunion(a:a, b, c...)
convexunion(a, b::Number, c...) = convexunion(a, b:b, c...)

_tounitrange(a) = first(a):last(a)

function colsupport(A::CachedMatrix, i)
    isempty(i) && return 1:0
    _tounitrange(minimum(i) ≤ A.datasize[2] ? convexunion(colsupport(A.array, i),colsupport(A.data,i) ∩ Base.OneTo(A.datasize[1])) : colsupport(A.array, i))
end
colsupport(A::CachedVector, i) =
    convexunion(colsupport(A.array, i),colsupport(A.data,i) ∩ Base.OneTo(A.datasize[1]))
function rowsupport(A::CachedMatrix, i)
    isempty(i) && return 1:0
    minimum(i) ≤ A.datasize[1] ? convexunion(rowsupport(A.array, i),rowsupport(A.data,i) ∩ Base.OneTo(A.datasize[2])) : rowsupport(A.array, i)
end


###
# special for zero cache
###
function cache_getindex(_, A::CachedVector{<:Any,<:AbstractVector,<:AbstractFill{<:Any,1}}, I::AbstractVector)
    @boundscheck checkbounds(A, I)
    CachedArray(A.data[I ∩ OneTo(A.datasize[1])], A.array[oneto(length(I))])
end

###
# MemoryLayout
####

struct CachedLayout{Data,Array} <: MemoryLayout end

cachedlayout(::Data, ::Array) where {Data,Array} = CachedLayout{Data,Array}()
MemoryLayout(C::Type{CachedArray{T,N,DAT,ARR}}) where {T,N,DAT,ARR} = cachedlayout(MemoryLayout(DAT), MemoryLayout(ARR))


#####
# broadcasting
#
# We want broadcasting for numbers with concatenations to pass through
# to take advantage of special implementations of the sub-components
######

BroadcastStyle(::Type{<:CachedArray{<:Any,N}}) where N = LazyArrayStyle{N}()

broadcasted(::LazyArrayStyle, op, A::CachedArray) = CachedArray(broadcast(op, cacheddata(A)), broadcast(op, A.array))
broadcasted(::LazyArrayStyle, op, A::CachedArray, c::Number) = CachedArray(broadcast(op, cacheddata(A), c), broadcast(op, A.array, c))
broadcasted(::LazyArrayStyle, op, c::Number, A::CachedArray) = CachedArray(broadcast(op, c, cacheddata(A)), broadcast(op, c, A.array))
broadcasted(::LazyArrayStyle, op, A::CachedArray, c::Ref) = CachedArray(broadcast(op, cacheddata(A), c), broadcast(op, A.array, c))
broadcasted(::LazyArrayStyle, op, c::Ref, A::CachedArray) = CachedArray(broadcast(op, c, cacheddata(A)), broadcast(op, c, A.array))


function layout_broadcasted(::CachedLayout, _, op, A::AbstractVector, B::AbstractVector)
    dat = cacheddata(A)
    n = length(dat)
    m = length(B)
    CachedArray(convert(Array, broadcast(op, dat, view(B,1:n))), broadcast(op, A.array, B))
end

function layout_broadcasted(_, ::CachedLayout, op, A::AbstractVector, B::AbstractVector)
    dat = cacheddata(B)
    n = length(dat)
    m = length(A)
    CachedArray(convert(Array, broadcast(op, view(A,1:n), dat)), broadcast(op, A, B.array))
end

function layout_broadcasted(::CachedLayout, ::CachedLayout, op, A::AbstractVector, B::AbstractVector)
    n = max(A.datasize[1],B.datasize[1])
    resizedata!(A,n)
    resizedata!(B,n)
    Adat = view(cacheddata(A),1:n)
    Bdat = view(cacheddata(B),1:n)
    CachedArray(convert(Array, broadcast(op, Adat, Bdat)), broadcast(op, A.array, B.array))
end

function layout_broadcasted(op, A, B)
    if axes(A) ≠ axes(B)
        (size(A,1) == 1 || size(B,1) == 1) && error("Internal error: Scalar-like broadcasting not yet supported.")
        throw(DimensionMismatch("arrays could not be broadcast to a common size; got a dimension with lengths $(length(A)) and $(length(B))"))
    end
    layout_broadcasted(MemoryLayout(A), MemoryLayout(B), op, A, B)
end

broadcasted(::LazyArrayStyle, op, A::CachedVector, B::AbstractVector) = layout_broadcasted(op, A, B)
broadcasted(::LazyArrayStyle, op, A::AbstractVector, B::CachedVector) = layout_broadcasted(op, A, B)
broadcasted(::LazyArrayStyle, op, A::CachedVector, B::CachedVector) = layout_broadcasted(op, A, B)
broadcasted(::LazyArrayStyle, op, A::Broadcasted, B::CachedVector) = broadcast(op, materialize(A), B)

broadcasted(::LazyArrayStyle, op, A::SubArray{<:Any,1,<:CachedMatrix}, B::CachedVector) = layout_broadcasted(op, A, B)
broadcasted(::LazyArrayStyle, op, A::SubArray{<:Any,1,<:CachedMatrix}, B::AbstractVector) = layout_broadcasted(op, A, B)
broadcasted(::LazyArrayStyle, op, A::CachedVector, B::SubArray{<:Any,1,<:CachedMatrix}) = layout_broadcasted(op, A, B)
broadcasted(::LazyArrayStyle, op, A::AbstractVector, B::SubArray{<:Any,1,<:CachedMatrix}) = layout_broadcasted(op, A, B)

broadcasted(::LazyArrayStyle{1}, op, a::CachedVector, b::Zeros{<:Any,1}) = broadcast(DefaultArrayStyle{1}(), op, a, b)
broadcasted(::LazyArrayStyle{1}, op, a::Zeros{<:Any,1}, b::CachedVector) = broadcast(DefaultArrayStyle{1}(), op, a, b)
broadcasted(::LazyArrayStyle{1}, ::typeof(*), a::CachedVector, b::Zeros{<:Any,1}) = broadcast(DefaultArrayStyle{1}(), *, a, b)
broadcasted(::LazyArrayStyle{1}, ::typeof(*), a::Zeros{<:Any,1}, b::CachedVector) = broadcast(DefaultArrayStyle{1}(), *, a, b)



###
# norm
###

# allow overloading for special backends, e.g., padded
_norm2(_, a) = sqrt(norm(cacheddata(a),2)^2 + norm(@view(a.array[a.datasize[1]+1:end]),2)^2)
_norm1(_, a) = norm(cacheddata(a),1) + norm(@view(a.array[a.datasize[1]+1:end]),1)
_normInf(_, a) = max(norm(cacheddata(a),Inf), norm(@view(a.array[a.datasize[1]+1:end]),Inf))
_normp(_, a, p) = (norm(cacheddata(a),p)^p + norm(@view(a.array[a.datasize[1]+1:end]),p)^p)^inv(p)

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

function fill!(a::SubArray{<:Any,1,<:CachedVector}, x)
    p = parent(a)
    kr = parentindices(a)[1]
    if !isempty(kr)
        N = maximum(kr)
        if isfinite(N)
            resizedata!(p, N)
            fill!(view(p.data, kr), x)
        else    
            fill!(view(p.data, kr ∩ OneTo(p.datasize[1])), x)
            fill!(view(p.array, kr), x)
        end
    end
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

###
# special maximum/minimum/sum
# TODO: when view returns a Fill this can be generalised
###


function maximum(a::CachedVector{<:Any,<:Any,<:AbstractFill})
    data = cacheddata(a)
    if length(data) < length(a)
        max(maximum(data), getindex_value(a.array))
    else
        maximum(data)
    end
end

function minimum(a::CachedVector{<:Any,<:Any,<:AbstractFill})
    data = cacheddata(a)
    if length(data) < length(a)
        min(minimum(data), getindex_value(a.array))
    else
        minimum(data)
    end
end

function sum(a::CachedVector{<:Any,<:Any,<:AbstractFill})
    data = cacheddata(a)
    sum(data) + sum(a.array[length(data)+1:end])
end


permutedims(a::CachedMatrix) = CachedArray(permutedims(a.data), permutedims(a.array), reverse(a.datasize))
permutedims(a::CachedVector) = CachedArray(permutedims(a.data), permutedims(a.array), (1,a.datasize[1]))


##
# hide type of AbstactArray
##
const CachedAbstractArray{T,N} = CachedArray{T,N,Array{T,N},AbstractArray{T,N}}
const CachedAbstractVector{T} = CachedAbstractArray{T,1}
const CachedAbstractMatrix{T} = CachedAbstractArray{T,2}

CachedAbstractArray{T,N}(data::AbstractArray{T,N}, array::AbstractArray{T,N}) where {T,N} = CachedAbstractArray{T,N}(data, array, size(data))
CachedAbstractArray{T,N}(array::AbstractArray{T,N}) where {T,N} = CachedAbstractArray{T,N}(Array{T,N}(undef, ntuple(zero,N)), array)
CachedAbstractArray(array::AbstractArray{T,N}) where {T,N} = CachedAbstractArray{T,N}(array)
CachedAbstractArray(data::AbstractArray{T,N}, array::AbstractArray{T,N}) where {T,N} = CachedAbstractArray{T,N}(data, array)

# TODO: uncomment when Julia v1.6 is required
# (CachedAbstractArray{V,N} where V)(array::AbstractArray{T,N}) where {T,N} = CachedAbstractArray{T,N}(array)
CachedAbstractVector(array::AbstractVector{T}) where T = CachedAbstractVector{T}(array)
CachedAbstractMatrix(array::AbstractMatrix{T}) where T = CachedAbstractMatrix{T}(array)


broadcasted(::LazyArrayStyle, op, A::CachedAbstractArray) = CachedAbstractArray(broadcast(op, cacheddata(A)), broadcast(op, A.array))
function broadcasted(::LazyArrayStyle, op, A::CachedAbstractVector, B::CachedAbstractVector)
    n = max(A.datasize[1],B.datasize[1])
    resizedata!(A,n)
    resizedata!(B,n)
    Adat = view(cacheddata(A),1:n)
    Bdat = view(cacheddata(B),1:n)
    CachedAbstractArray(broadcast(op, Adat, Bdat), broadcast(op, A.array, B.array))
end
broadcasted(::LazyArrayStyle, op, A::CachedAbstractArray, c::Number) = CachedAbstractArray(broadcast(op, cacheddata(A), c), broadcast(op, A.array, c))
broadcasted(::LazyArrayStyle, op, c::Number, A::CachedAbstractArray) = CachedAbstractArray(broadcast(op, c, cacheddata(A)), broadcast(op, c, A.array))
broadcasted(::LazyArrayStyle, op, A::CachedAbstractArray, c::Ref) = CachedAbstractArray(broadcast(op, cacheddata(A), c), broadcast(op, A.array, c))
broadcasted(::LazyArrayStyle, op, c::Ref, A::CachedAbstractArray) = CachedAbstractArray(broadcast(op, c, cacheddata(A)), broadcast(op, c, A.array))


###
# copyto!
###

function copyto!_layout(_, ::CachedLayout, dest::AbstractArray{T,N}, src::AbstractArray{V,N}) where {T,V,N}
    resizedata!(src, size(dest)...)
    copyto!(dest, view(cacheddata(src), axes(dest)...))
end

###
# SubArray
###

sublayout(::CachedLayout{MLAY,ALAY}, ::Type{I}) where {MLAY,ALAY,I} =
    cachedlayout(sublayout(MLAY(),I), sublayout(ALAY,I))

function resizedata!(V::SubArray, n::Integer...)
    resizedata!(parent(V), getindex.(parentindices(V), max.(1,n))...)
    V
end

function resizedata!(v::SubArray{<:Any,1,<:AbstractMatrix,<:Tuple{AbstractVector,Integer}}, m::Integer)
    kr,j = parentindices(v)
    resizedata!(parent(v), maximum(view(kr,1:m)), j)
    v
end

function resizedata!(v::SubArray{<:Any,1,<:AbstractMatrix,<:Tuple{Integer,AbstractVector}}, m::Integer)
    k,jr = parentindices(v)
    resizedata!(parent(v), k, maximum(view(jr,1:m)))
    v
end

function cacheddata(V::SubArray)
    P = parent(V)
    data = cacheddata(P)
    view(data, intersect.(axes(data), parentindices(V))...)
end


##
# AbstractQ
##
cache(A::AbstractQ) = _cache(MemoryLayout(A), A)
cache_layout(_, O::AbstractQ) = CachedArray(O)
CachedArray(array::AbstractQ{T}) where T = CachedArray(similar(Matrix{T}, (0,0)), array)
CachedArray(data::AbstractMatrix, array::AbstractQ) = CachedArray(data, array, size(data))
CachedArray(data::AbstractMatrix{T}, array::AbstractQ{T}, datasize::NTuple{2,Int}) where T =
    CachedMatrix{T,typeof(data),typeof(array)}(data, array, datasize)

length(A::CachedMatrix{<:T,<:AbstractMatrix{T},<:AbstractQ{T}}) where T = prod(size(A.array))