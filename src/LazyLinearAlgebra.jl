module LazyLinearAlgebra
using Base
using Base.Broadcast
using LinearAlgebra
using LinearAlgebra.BLAS

import Base: ReinterpretArray, ReshapedArray, AbstractCartesianIndex, Slice,
             RangeIndex, BroadcastStyle, copyto!, length, broadcastable, axes
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

const MatrixMulVector{T, styleA, styleX, AType<:AbstractMatrix, XType<:AbstractVector} = Mul{T, styleA, styleX, AType, XType}

length(M::MatrixMulVector) = size(M.A,1)
axes(M::MatrixMulVector) = (Base.OneTo(length(M)),)
broadcastable(M::MatrixMulVector) = M


BroadcastStyle(::Type{<:MatrixMulVector{<:Any,StyleA,StyleX}}) where {StyleA,StyleX} = MatrixMulVectorStyle{StyleA,StyleX}()
BroadcastStyle(M::MatrixMulVectorStyle, ::DefaultArrayStyle) = M
BroadcastStyle(::DefaultArrayStyle, M::MatrixMulVectorStyle) = M
similar(M::Broadcasted{<:MatrixMulVectorStyle}, ::Type{ElType}) where ElType = Vector{Eltype}(length(M.args[1]))

copyto!(dest::AbstractVector, bc::Broadcasted{<:MatrixMulVectorStyle}) =
    _copyto!(MemoryLayout(dest), dest, bc)

_copyto!(::AbstractStridedLayout, dest::AbstractVector,
         bc::Broadcasted{<:MatrixMulVectorStyle{<:AbstractColumnMajor,<:AbstractStridedLayout},
                         Tuple{Base.OneTo{Int}}, typeof(identity),
                         <:Tuple{<:MatrixMulVector{T}}}) where T<: BlasFloat =
    BLAS.gemv!('N', one(T), bc.args[1].A, bc.args[1].x, zero(T), dest)


_copyto!(::AbstractStridedLayout, dest::AbstractVector,
         bc::Broadcasted{<:MatrixMulVectorStyle{<:AbstractColumnMajor,<:AbstractStridedLayout},
                         Tuple{Base.OneTo{Int}}, typeof(*),
                         <:Tuple{T,<:MatrixMulVector{T}}}) where T<: BlasFloat =
    BLAS.gemv!('N', bc.args[1], bc.args[2].A, bc.args[2].x, zero(T), dest)


function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
         bc::Broadcasted{<:MatrixMulVectorStyle{<:AbstractColumnMajor,<:AbstractStridedLayout},
                         Tuple{Base.OneTo{Int}}, typeof(+),
                         <:Tuple{<:MatrixMulVector{T},<:Vector{T}}}) where T<: BlasFloat
    c = bc.args[2]
    c ≡ dest || copyto!(dest, c)
    BLAS.gemv!('N', one(T), bc.args[1].A, bc.args[1].x, one(T), dest)
end

function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
         bc::Broadcasted{<:MatrixMulVectorStyle{<:AbstractColumnMajor,<:AbstractStridedLayout},
                         Tuple{Base.OneTo{Int}}, typeof(+),
                         <:Tuple{<:MatrixMulVector{T},
                                 Broadcasted{DefaultArrayStyle{1},Nothing,typeof(*),Tuple{T,Vector{T}}}}}) where T<: BlasFloat
    c = bc.args[2].args[2]
    c ≡ dest || copyto!(dest, c)
    BLAS.gemv!('N', one(T), bc.args[1].A, bc.args[1].x, bc.args[2].args[1], dest)
end

function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
         bc::Broadcasted{<:MatrixMulVectorStyle{<:AbstractColumnMajor,<:AbstractStridedLayout},
                         Tuple{Base.OneTo{Int}}, typeof(+),
                         <:Tuple{Broadcasted{<:MatrixMulVectorStyle{<:AbstractColumnMajor,<:AbstractStridedLayout},
                                             Nothing, typeof(*), <:Tuple{T,<:MatrixMulVector{T}}},
                                 Broadcasted{DefaultArrayStyle{1},Nothing,typeof(*),Tuple{T,Vector{T}}}}}) where T<: BlasFloat
    c = bc.args[2].args[2]
    c ≡ dest || copyto!(dest, c)
    BLAS.gemv!('N', bc.args[1].args[1], bc.args[1].args[2].A, bc.args[1].args[2].x, bc.args[2].args[1], dest)
end

end # module
