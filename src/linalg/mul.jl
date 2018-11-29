
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
axes(M::Mul{Tuple{}}) = ()

similar(M::Mul) = similar(M, eltype(M))



"""
   lmaterialize(M::Mul)

materializes arrays iteratively, left-to-right.
"""

_lmaterialize(A, B) = materialize(Mul((A,B)))
_lmaterialize(A, B, C, D...) = _lmaterialize(materialize(Mul((A,B))), C, D...)

lmaterialize(M::Mul) = _lmaterialize(M.factors...)

_rmaterialize(Z, Y) = materialize(Mul((Y,Z)))
_rmaterialize(Z, Y, X, W...) = _rmaterialize(materialize(Mul((Y,Z))), X, W...)

rmaterialize(M::Mul) = _rmaterialize(reverse(M.factors)...)


*(A::Mul, B::Mul) = materialize(Mul(A.factors..., B.factors...))
*(A::Mul, B) = materialize(Mul(A.factors..., B))
*(A, B::Mul) = materialize(Mul(A, B.factors...))
⋆(A...) = Mul(A...)


####
# Matrix * Array
####

const ArrayMulArray{styleA, styleB, p, q, T, V} =
    Mul2{styleA, styleB, <:AbstractArray{T,p}, <:AbstractArray{V,q}}

const ArrayMuls = Mul{<:Tuple, <:Tuple{Vararg{<:AbstractArray}}}

# the default is always Array
similar(M::ArrayMuls, ::Type{T}, ::NTuple{N,OneTo{Int}}) where {T,N} = Array{T}(undef, size(M))
similar(M::ArrayMuls, ::Type{T}) where T = similar(M, T, axes(M))
_materialize(M::ArrayMulArray, _) = copyto!(similar(M), M)
_materialize(M::ArrayMuls, _) = lmaterialize(M)
_materialize(M::Mul, _) = lmaterialize(M)
_materialize(M::Mul2, _) = error("Cannot materialize $M")
materialize(M::Mul) = _materialize(M, axes(M))

@inline copyto!(dest::AbstractArray, M::Mul) = _copyto!(MemoryLayout(dest), dest, M)


####
# Matrix * Vector
####
const MatMulVec{styleA, styleB, T, V} = ArrayMulArray{styleA, styleB, 2, 1, T, V}


rowsupport(_, A, k) = axes(A,2)
""""
    rowsupport(A, k)

gives an iterator containing the possible non-zero entries in the k-th row of A.
"""
rowsupport(A, k) = rowsupport(MemoryLayout(A), A, k)

colsupport(_, A, j) = axes(A,1)

""""
    colsupport(A, j)

gives an iterator containing the possible non-zero entries in the j-th column of A.
"""
colsupport(A, j) = colsupport(MemoryLayout(A), A, j)


function getindex(M::MatMulVec, k::Integer)
    A,B = M.factors
    ret = zero(eltype(M))
    for j = rowsupport(A, k)
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
    @inbounds for ℓ in (rowsupport(A,k) ∩ colsupport(B,j))
        ret += A[k,ℓ] * B[ℓ,j]
    end
    ret
end

getindex(M::MatMulMat, kj::CartesianIndex{2}) = M[kj[1], kj[2]]


####
# MulArray
#####

function getindex(M::Mul, k)
    A,Bs = first(M.factors), tail(M.factors)
    B = Mul(Bs)
    ret = zero(eltype(M))
    for j = rowsupport(A, k)
        ret += A[k,j] * B[j]
    end
    ret
end

function getindex(M::Mul, k, j)
    A,Bs = first(M.factors), tail(M.factors)
    B = Mul(Bs)
    ret = zero(eltype(M))
    @inbounds for ℓ in (rowsupport(A,k) ∩ colsupport(B,j))
        ret += A[k,ℓ] * B[ℓ,j]
    end
    ret
end


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
*(A::MulArray, B::Mul) = A.mul * B
*(A::Mul, B::MulArray) = A * B.mul

adjoint(A::MulArray) = MulArray(reverse(adjoint.(A.mul.factors))...)
transpose(A::MulArray) = MulArray(reverse(transpose.(A.mul.factors))...)


struct MulLayout{LAY} <: MemoryLayout
    layouts::LAY
end

MemoryLayout(M::MulArray) = MulLayout(MemoryLayout.(M.mul.factors))


_flatten(A::MulArray, B...) = _flatten(A.mul.factors..., B...)
flatten(A::MulArray) = MulArray(Mul(_flatten(A.mul.factors...)))
