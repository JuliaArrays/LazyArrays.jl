module LazyLinearAlgebra
using Base
using Base.Broadcast
using LinearAlgebra
using LinearAlgebra.BLAS

import Base: ReinterpretArray, ReshapedArray, AbstractCartesianIndex, Slice,
             RangeIndex, BroadcastStyle, copyto!, length, broadcastable, axes,
             getindex, eltype
import Base.Broadcast: DefaultArrayStyle, Broadcasted
import LinearAlgebra.BLAS: BlasFloat, BlasReal, BlasComplex
export Mul

include("memorylayout.jl")

struct MatrixMulVectorStyle{StyleA, StyleX} <: BroadcastStyle end

struct Mul{T, StyleA, StyleX, AType, XType}
    style_A::StyleA
    style_x::StyleX
    A::AType
    x::XType
end


Mul(styleA, stylex, A, x) =
    Mul{Base.promote_op(*, eltype(A), eltype(x)),
        typeof(styleA), typeof(stylex),
        typeof(A), typeof(x)}(styleA, stylex, A, x)
Mul(A, x) = Mul(MemoryLayout(A), MemoryLayout(x), A, x)

eltype(::Mul{T}) where T = T

const MatrixMulVector{T, styleA, styleX, AType<:AbstractMatrix, XType<:AbstractVector} =
    Mul{T, styleA, styleX, AType, XType}
const BMatVec{T, styleA, styleB} =
    Broadcasted{<:MatrixMulVectorStyle{styleA,styleB},
                Tuple{Base.OneTo{Int}}, typeof(identity),
                <:Tuple{<:MatrixMulVector{T}}}
const BConstMatVec{T,styleA,styleB} =
    Broadcasted{<:MatrixMulVectorStyle{styleA,styleB},
                    Tuple{Base.OneTo{Int}}, typeof(*),
                    <:Tuple{T,<:MatrixMulVector{T}}}
const BMatVecPlusVec{T,styleA,styleB} =
    Broadcasted{<:MatrixMulVectorStyle{styleA, styleB},
                Tuple{Base.OneTo{Int}}, typeof(+),
                <:Tuple{<:MatrixMulVector{T},<:Vector{T}}}
const BMatVecPlusConstVec{T,styleA,styleB} =
    Broadcasted{<:MatrixMulVectorStyle{styleA,styleB},
                Tuple{Base.OneTo{Int}}, typeof(+),
                <:Tuple{<:MatrixMulVector{T},
                        Broadcasted{DefaultArrayStyle{1},Nothing,typeof(*),Tuple{T,Vector{T}}}}}
const BConstMatVecPlusConstVec{T, styleA, styleB} =
    Broadcasted{<:MatrixMulVectorStyle{styleA,styleB},
                Tuple{Base.OneTo{Int}}, typeof(+),
                <:Tuple{Broadcasted{<:MatrixMulVectorStyle{styleA,styleB},
                                    Nothing, typeof(*), <:Tuple{T,<:MatrixMulVector{T}}},
                        Broadcasted{DefaultArrayStyle{1},Nothing,typeof(*),Tuple{T,Vector{T}}}}}

length(M::MatrixMulVector) = size(M.A,1)
axes(M::MatrixMulVector) = (Base.OneTo(length(M)),)
broadcastable(M::MatrixMulVector) = M

function getindex(M::MatrixMulVector{T}, k::Int) where T
    ret = zero(T)
    for j = 1:size(M.A,2)
        ret += M.A[k,j] * M.x[j]
    end
    ret
end

getindex(M::MatrixMulVector, k::CartesianIndex{1}) = M[convert(Int, k)]

BroadcastStyle(::Type{<:MatrixMulVector{<:Any,StyleA,StyleX}}) where {StyleA,StyleX} = MatrixMulVectorStyle{StyleA,StyleX}()
BroadcastStyle(M::MatrixMulVectorStyle, ::DefaultArrayStyle) = M
BroadcastStyle(::DefaultArrayStyle, M::MatrixMulVectorStyle) = M
similar(M::Broadcasted{<:MatrixMulVectorStyle}, ::Type{ElType}) where ElType = Vector{Eltype}(length(M.args[1]))

copyto!(dest::AbstractVector, bc::Broadcasted{<:MatrixMulVectorStyle}) =
    _copyto!(MemoryLayout(dest), dest, bc)

# Use default
_copyto!(_, dest, bc) = copyto!(dest, Broadcasted{Nothing}(bc.f, bc.args, bc.axes))

_copyto!(::AbstractStridedLayout, dest::AbstractVector,
         bc::BMatVec{T, <:AbstractColumnMajor, <:AbstractStridedLayout}) where T<: BlasFloat =
    BLAS.gemv!('N', one(T), bc.args[1].A, bc.args[1].x, zero(T), dest)



_copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         bc::BConstMatVec{T, <:AbstractColumnMajor,<:AbstractStridedLayout}) where T<: BlasFloat =
    BLAS.gemv!('N', bc.args[1], bc.args[2].A, bc.args[2].x, zero(T), dest)


function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
         bc::BMatVecPlusVec{T,<:AbstractColumnMajor,<:AbstractStridedLayout}) where T<: BlasFloat
    c = bc.args[2]
    c ≡ dest || copyto!(dest, c)
    BLAS.gemv!('N', one(T), bc.args[1].A, bc.args[1].x, one(T), dest)
end

function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
         bc::BMatVecPlusConstVec{T,<:AbstractColumnMajor,<:AbstractStridedLayout}) where T<: BlasFloat
    c = bc.args[2].args[2]
    c ≡ dest || copyto!(dest, c)
    BLAS.gemv!('N', one(T), bc.args[1].A, bc.args[1].x, bc.args[2].args[1], dest)
end


function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
         bc::BConstMatVecPlusConstVec{T,<:AbstractColumnMajor,<:AbstractStridedLayout}) where T<: BlasFloat
    c = bc.args[2].args[2]
    c ≡ dest || copyto!(dest, c)
    BLAS.gemv!('N', bc.args[1].args[1], bc.args[1].args[2].A, bc.args[1].args[2].x, bc.args[2].args[1], dest)
end

end # module
