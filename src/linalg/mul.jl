
checkdimensions() = nothing
checkdimensions(_) = nothing

function checkdimensions(A, B, C...)
    axes(A,2) == axes(B,1) || throw(DimensionMismatch(""))
    checkdimensions(B, C...)
end

struct Mul{Styles<:Tuple, Factors<:Tuple}
    styles::Styles
    factors::Factors
    function Mul{S,F}(styles::S, factors::F) where {S,F}
        checkdimensions(factors...)
        new{S,F}(styles,factors)
    end
end

Mul(styles::S, factors::F) where {S<:Tuple,F<:Tuple} = Mul{S,F}(styles, factors)

Mul(A::Tuple) = Mul(MemoryLayout.(A), A)

"""
 Mul(A1, A2, …, AN)

represents lazy multiplication A1*A2*…*AN. The factors must have compatible axes.
If any argument is itself a Mul, it automatically gets flatten. That is,
we assume associativity. Use Mul((A, B, C)) to stop flattening
"""
Mul(A...) = flatten(Mul(A))

_flatten() = ()
_flatten(A, B...) = (A, _flatten(B...)...)
_flatten(A::Mul, B...) = _flatten(A.factors..., B...)
flatten(A::Mul) = Mul(_flatten(A.factors...))


const Mul2{StyleA, StyleB, AType, BType} = Mul{<:Tuple{StyleA,StyleB}, <:Tuple{AType,BType}}

_mul_eltype(a) = eltype(a)
_mul_eltype(a, b...) = Base.promote_op(*, eltype(a), _mul_eltype(b...))
eltype(M::Mul) = _mul_eltype(M.factors...)
size(M::Mul, p::Int) = size(M)[p]
axes(M::Mul, p::Int) = axes(M)[p]
ndims(M::Mul) = ndims(last(M.factors))

length(M::Mul) = prod(size(M))
size(M::Mul) = length.(axes(M))

_mul_axes(ax1, ::Tuple{}) = (ax1,)
_mul_axes(ax1, ::Tuple{<:Any}) = (ax1,)
_mul_axes(ax1, (_,ax2)::Tuple{<:Any,<:Any}) = (ax1,ax2)
axes(M::Mul) = _mul_axes(axes(first(M.factors),1), axes(last(M.factors)))

similar(M::Mul) = similar(M, eltype(M))


# default is to stay Lazy
materialize(M::Mul2) = M


# re-materialize if the mul actually changed the type of Y, otherwise leave as a Mul
_rmaterialize_if_changed(::S, Z, Y::S, X...) where S = Mul(reverse(X)..., Y, Z)
_rmaterialize_if_changed(::S, Z::Mul, Y::S, X...) where S = Mul(reverse(X)..., Y, Z.factors...)
_rmaterialize_if_changed(_, Z, Y, X...) = _recursive_rmaterialize(Z, Y, X...)
_rmaterialize_if_changed(Y_old, Z, Y_new::Mul) = _rmaterialize_if_changed(Y_old, Z, reverse(Y_new.factors)...)

# materialize but get rid of Muls
_flatten_rmaterialize(A...) = _recursive_rmaterialize(A...)
function _flatten_rmaterialize(Z::Mul, Y...)
    tl = tail(reverse(Z.factors))
    _rmaterialize_if_changed(first(tl), last(Z.factors), _recursive_rmaterialize(tl..., Y...))
end

# repeatedly try to materialize two terms at a time
_recursive_rmaterialize(Z) = materialize(Z)
_recursive_rmaterialize(Z, Y) = Y*Z
_recursive_rmaterialize(Z, Y, X, W...) = _flatten_rmaterialize(Y*Z, X, W...)

"""
   rmaterialize(M::Mul)

materializes arrays iteratively, right-to-left.
"""

rmaterialize(M::Mul) = _recursive_rmaterialize(reverse(M.factors)...)

materialize(M::Mul) = rmaterialize(M)

*(A::Mul, B::Mul) = materialize(Mul(A.factors..., B.factors...))
*(A::Mul, B) = materialize(Mul(A.factors..., B))
*(A, B::Mul) = materialize(Mul(A, B.factors...))


####
# Matrix * Array
####

const ArrayMulArray{styleA, styleB, p, q, T, V} =
    Mul2{styleA, styleB, <:AbstractArray{T,p}, <:AbstractArray{V,q}}

const ArrayMuls = Mul{<:Tuple, <:Tuple{Vararg{<:AbstractArray}}}

# the default is always Array
similar(M::ArrayMuls, ::Type{T}, ::NTuple{N,OneTo{Int}}) where {T,N} = Array{T}(undef, size(M))
similar(M::ArrayMuls, ::Type{T}) where T = similar(M, T, axes(M))
_materialize(M::ArrayMuls, _) = copyto!(similar(M), M)
materialize(M::ArrayMuls) = _materialize(M, axes(M))

@inline copyto!(dest::AbstractArray, M::Mul) = _copyto!(MemoryLayout(dest), dest, M)


####
# Matrix * Vector
####
const MatMulVec{styleA, styleB, T, V} = ArrayMulArray{styleA, styleB, 2, 1, T, V}

function getindex(M::MatMulVec, k::Integer)
    A,B = M.factors
    ret = zero(eltype(M))
    for j = 1:size(A,2)
        ret += A[k,j] * B[j]
    end
    ret
end

getindex(M::MatMulVec, k::CartesianIndex{1}) = M[convert(Int, k)]



####
# Matrix * Matrix
####

const MatMulMat{styleA, styleB, T, V} = ArrayMulArray{styleA, styleB, 2, 2, T, V}

function getindex(M::MatMulMat, k::Integer, j::Integer)
    A,B = M.factors
    ret = zero(eltype(M))
    @inbounds for ℓ in axes(A,2)
        ret += A[k,ℓ] * B[ℓ,j]
    end
    ret
end

getindex(M::MatMulMat, kj::CartesianIndex{2}) = M[kj[1], kj[2]]


####
# MulArray
#####

struct MulArray{T, N, MUL<:Mul} <: AbstractArray{T,N}
    mul::MUL
end

const MulVector{T, MUL<:Mul} = MulArray{T, 1, MUL}
const MulMatrix{T, MUL<:Mul} = MulArray{T, 2, MUL}


MulArray{T,N}(M::MUL) where {T,N,MUL<:Mul} = MulArray{T,N,MUL}(M)
MulArray{T}(M::Mul) where {T} = MulArray{T,ndims(M)}(M)
MulArray(M::Mul) = MulArray{eltype(M)}(M)
MulVector(M::Mul) = MulVector{eltype(M)}(M)
MulMatrix(M::Mul) = MulMatrix{eltype(M)}(M)

MulArray(factors...) = MulArray(Mul(factors...))
MulArray{T}(factors...) where T = MulArray{T}(Mul(factors...))
MulArray{T,N}(factors...) where {T,N} = MulArray{T,N}(Mul(factors...))
MulVector(factors...) = MulVector(Mul(factors...))
MulMatrix(factors...) = MulMatrix(Mul(factors...))

axes(A::MulArray) = axes(A.mul)
size(A::MulArray) = map(length, axes(A))

IndexStyle(::MulArray{<:Any,1}) = IndexLinear()

@propagate_inbounds getindex(A::MulArray, kj::Int...) = A.mul[kj...]

*(A::MulArray, B::MulArray) = A.mul * B.mul


struct MulLayout{LAY} <: MemoryLayout
    layouts::LAY
end

MemoryLayout(M::MulArray) = MulLayout(MemoryLayout.(M.mul.factors))
