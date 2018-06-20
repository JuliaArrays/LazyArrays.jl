

struct Mul{T, StyleA, StyleB, AType, BType}
    style_A::StyleA
    style_B::StyleB
    A::AType
    B::BType
end


Mul(styleA, stylex, A, x) =
    Mul{Base.promote_op(*, eltype(A), eltype(x)),
        typeof(styleA), typeof(stylex),
        typeof(A), typeof(x)}(styleA, stylex, A, x)
Mul(A, x) = Mul(MemoryLayout(A), MemoryLayout(x), A, x)

eltype(::Mul{T}) where T = T

####
# Matrix * Array
####

struct ArrayMulArrayStyle{StyleA, StyleB, p, q} <: BroadcastStyle end
const ArrayMulArray{T, styleA, styleB, p, q} =
    Mul{T, styleA, styleB, <:AbstractArray{T,p}, <:AbstractArray{T,q}}

const BArrayMulArray{T, styleA, styleB, p, q} =
    Broadcasted{ArrayMulArrayStyle{styleA,styleB,p,q}, <:Any, typeof(identity),
                <:Tuple{<:ArrayMulArray{T,styleA,styleB,p,q}}}
const BConstArrayMulArray{T, styleA, styleB, p, q} =
    Broadcasted{ArrayMulArrayStyle{styleA,styleB,p,q},
                    <:Any, typeof(*),
                    <:Tuple{T,<:ArrayMulArray{T,styleA,styleB,p,q}}}
const BArrayMulArrayPlusArray{T, styleA, styleB, p, q} =
    Broadcasted{ArrayMulArrayStyle{styleA,styleB,p,q},
                <:Any, typeof(+),
                <:Tuple{<:ArrayMulArray{T,styleA,styleB,p,q},<:AbstractArray{T,q}}}
const BArrayMulArrayPlusConstArray{T, styleA, styleB, p, q} =
    Broadcasted{ArrayMulArrayStyle{styleA,styleB,p,q},
                <:Any, typeof(+),
                <:Tuple{<:ArrayMulArray{T,styleA,styleB,p,q},
                Broadcasted{DefaultArrayStyle{q},<:Any,typeof(*),
                            <:Tuple{T,<:AbstractArray{T,q}}}}}
const BConstArrayMulArrayPlusArray{T, styleA, styleB, p, q} =
    Broadcasted{ArrayMulArrayStyle{styleA,styleB,p,q},
                <:Any, typeof(+),
                <:Tuple{Broadcasted{<:ArrayMulArrayStyle{styleA,styleB,p,q},
                                    <:Any, typeof(*),
                                    <:Tuple{T,<:ArrayMulArray{T,styleA,styleB,p,q}}},
                        <:AbstractArray{T,q}}}
const BConstArrayMulArrayPlusConstArray{T, styleA, styleB, p, q} =
    Broadcasted{ArrayMulArrayStyle{styleA,styleB,p,q},
                <:Any, typeof(+),
                <:Tuple{Broadcasted{<:ArrayMulArrayStyle{styleA,styleB,p,q},
                                    <:Any, typeof(*),
                                    <:Tuple{T,<:ArrayMulArray{T,styleA,styleB,p,q}}},
                        Broadcasted{DefaultArrayStyle{q},<:Any,typeof(*),<:Tuple{T,<:AbstractArray{T,q}}}}}


BroadcastStyle(::Type{<:ArrayMulArray{<:Any,StyleA,StyleB,p,q}}) where {StyleA,StyleB,p,q} = ArrayMulArrayStyle{StyleA,StyleB,p,q}()
BroadcastStyle(M::ArrayMulArrayStyle, ::DefaultArrayStyle) = M
BroadcastStyle(::DefaultArrayStyle, M::ArrayMulArrayStyle) = M
similar(M::Broadcasted{<:ArrayMulArrayStyle}, ::Type{ElType}) where ElType = Array{Eltype}(undef,size(M.args[1]))

@inline copyto!(dest::AbstractArray, bc::Broadcasted{<:ArrayMulArrayStyle}) =
    _copyto!(MemoryLayout(dest), dest, bc)
####
# Matrix * Vector
####
let (p,q) = (2,1)
    global const MatMulVecStyle{StyleA, StyleB} = ArrayMulArrayStyle{StyleA, StyleB, p, q}
    global const MatMulVec{T, styleA, styleB} = ArrayMulArray{T, styleA, styleB, p, q}

    global const BMatVec{T, styleA, styleB} = BArrayMulArray{T, styleA, styleB, p, q}
    global const BConstMatVec{T, styleA, styleB} = BConstArrayMulArray{T, styleA, styleB, p, q}
    global const BMatVecPlusVec{T,styleA,styleB} = BArrayMulArrayPlusArray{T, styleA, styleB, p, q}
    global const BMatVecPlusConstVec{T,styleA,styleB} = BArrayMulArrayPlusConstArray{T, styleA, styleB, p, q}
    global const BConstMatVecPlusVec{T, styleA, styleB} = BConstArrayMulArrayPlusArray{T, styleA, styleB, p, q}
    global const BConstMatVecPlusConstVec{T, styleA, styleB} = BConstArrayMulArrayPlusConstArray{T, styleA, styleB, p, q}
end


length(M::MatMulVec) = size(M.A,1)
axes(M::MatMulVec) = (axes(M.A,1),)
broadcastable(M::MatMulVec) = M
instantiate(bc::Broadcasted{<:MatMulVecStyle}) = bc

# function getindex(M::MatMulVec{T}, k::Int) where T
#     ret = zero(T)
#     for j = 1:size(M.A,2)
#         ret += M.A[k,j] * M.B[j]
#     end
#     ret
# end

getindex(M::MatMulVec, k::CartesianIndex{1}) = M[convert(Int, k)]

# Use default
# @inline _copyto!(_, dest, bc) = copyto!(dest, Broadcasted{Nothing}(bc.f, bc.args, bc.axes))

# Matrix * Vector

# @inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
#          bc::BMatVec{T, <:AbstractColumnMajor, <:AbstractStridedLayout}) where T<: BlasFloat
#     (M,) = bc.args
#     dest .= one(T) .* M .+ zero(T) .* dest
# end

# make copy to make sure always works
@inline function _gemv!(tA, α, A, x, β, y)
    if x ≡ y
        BLAS.gemv!(tA, α, A, copy(x), β, y)
    else
        BLAS.gemv!(tA, α, A, x, β, y)
    end
end

@inline function _gemv!(dest, trans, α, A, x, β, y)
    y ≡ dest || copyto!(dest, y)
    _gemv!(trans, α, A, x, β, dest)
end

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
         bc::BMatVec{T, <:AbstractColumnMajor, <:AbstractStridedLayout}) where T<: BlasFloat
    (M,) = bc.args
    A,x = M.A, M.B
    _gemv!('N', one(T), A, x, zero(T), dest)
end


@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         bc::BConstMatVec{T, <:AbstractColumnMajor,<:AbstractStridedLayout}) where T<: BlasFloat
    α,M = bc.args
    A,x = M.A, M.B
    _gemv!('N', α, A, x, zero(T), dest)
end


@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
         bc::BMatVecPlusVec{T,<:AbstractColumnMajor,<:AbstractStridedLayout}) where T<: BlasFloat
    M,y = bc.args
    A,x = M.A, M.B
    _gemv!(dest, 'N', one(T), A, x, one(T), y)
end

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
         bc::BMatVecPlusConstVec{T,<:AbstractColumnMajor,<:AbstractStridedLayout}) where T<: BlasFloat
    M,βc = bc.args
    β,y = βc.args
    A,x = M.A, M.B
    _gemv!(dest, 'N', one(T), A, x, β, y)
end


@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
         bc::BConstMatVecPlusVec{T,<:AbstractColumnMajor,<:AbstractStridedLayout}) where T<: BlasFloat
    αM,y = bc.args
    α,M = αM.args
    A,x = M.A, M.B
    _gemv!(dest, 'N', α, A, x, one(T), y)
end


@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
         bc::BConstMatVecPlusConstVec{T,<:AbstractColumnMajor,<:AbstractStridedLayout}) where T<: BlasFloat
    αM,βc = bc.args
    α,M = αM.args
    A,x = M.A, M.B
    β,y = βc.args
    _gemv!(dest, 'N', α, A, x, β, y)
end

# AdjTrans

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
         bc::BConstMatVecPlusConstVec{T,<:AbstractRowMajor,<:AbstractStridedLayout}) where T<: BlasReal
    αM,βc = bc.args
    α,M = αM.args
    A,x = M.A, M.B
    β,y = βc.args
    _gemv!(dest, 'T', α, transpose(A), x, β, y)
end

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
         bc::BConstMatVecPlusConstVec{T,<:AbstractRowMajor,<:AbstractStridedLayout}) where T<: BlasComplex
    αM,βc = bc.args
    α,M = αM.args
    A,x = M.A, M.B
    β,y = βc.args
    _gemv!(dest, 'C', α, A', x, β, y)
end


####
# Matrix * Matrix
####


let (p,q) = (2,2)
    global const MatMulMatStyle{StyleA, StyleB} = ArrayMulArrayStyle{StyleA, StyleB, p, q}
    global const MatMulMat{T, styleA, styleB} = ArrayMulArray{T, styleA, styleB, p, q}

    global const BMatMat{T, styleA, styleB} = BArrayMulArray{T, styleA, styleB, p, q}
    global const BConstMatMat{T, styleA, styleB} = BConstArrayMulArray{T, styleA, styleB, p, q}
    global const BMatMatPlusMat{T,styleA,styleB} = BArrayMulArrayPlusArray{T, styleA, styleB, p, q}
    global const BMatMatPlusConstMat{T,styleA,styleB} = BArrayMulArrayPlusConstArray{T, styleA, styleB, p, q}
    global const BConstMatMatPlusMat{T, styleA, styleB} = BConstArrayMulArrayPlusArray{T, styleA, styleB, p, q}
    global const BConstMatMatPlusConstMat{T, styleA, styleB} = BConstArrayMulArrayPlusConstArray{T, styleA, styleB, p, q}
end


size(M::MatMulMat) = (size(M.A,1),size(M.B,2))
axes(M::MatMulMat) = (axes(M.A,1),axes(M.B,2))
broadcastable(M::MatMulMat) = M
instantiate(bc::Broadcasted{<:MatMulMatStyle}) = bc

# function getindex(M::MatMulVec{T}, k::Int) where T
#     ret = zero(T)
#     for ℓ in axes(M.A,2)
#         ret += M.A[k,ℓ] * M.B[ℓ,j]
#     end
#     ret
# end

getindex(M::MatMulMat, kj::CartesianIndex{2}) = M[kj...]



# Use default
# @inline _copyto!(_, dest, bc) = copyto!(dest, Broadcasted{Nothing}(bc.f, bc.args, bc.axes))

# Matrix * Vector

# @inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
#          bc::BMatVec{T, <:AbstractColumnMajor, <:AbstractStridedLayout}) where T<: BlasFloat
#     (M,) = bc.args
#     dest .= one(T) .* M .+ zero(T) .* dest
# end

# make copy to make sure always works
@inline function _gemm!(tA, tB, α, A, B, β, C)
    if B ≡ C
        BLAS.gemm!(tA, tB, α, A, copy(B), β, C)
    else
        BLAS.gemm!(tA, tB, α, A, B, β, C)
    end
end

@inline function _gemm!(dest, tA, tB, α, A, B, β, C)
    C ≡ dest || copyto!(dest, C)
    _gemm!(tA, tB, α, A, B, β, dest)
end

@inline function _copyto!(::AbstractColumnMajor, dest::AbstractMatrix,
         bc::BMatMat{T, <:AbstractColumnMajor, <:AbstractColumnMajor}) where T<: BlasFloat
    (M,) = bc.args
    A,B = M.A, M.B
    _gemm!('N', 'N', one(T), A, B, zero(T), dest)
end


@inline function _copyto!(::AbstractColumnMajor, dest::AbstractMatrix{T},
         bc::BConstMatMat{T, <:AbstractColumnMajor,<:AbstractColumnMajor}) where T<: BlasFloat
    α,M = bc.args
    A,B = M.A, M.B
    _gemm!('N', 'N', α, A, B, zero(T), dest)
end


@inline function _copyto!(::AbstractColumnMajor, dest::AbstractMatrix,
         bc::BMatMatPlusMat{T,<:AbstractColumnMajor,<:AbstractColumnMajor}) where T<: BlasFloat
    M,C = bc.args
    A,B = M.A, M.B
    _gemm!(dest, 'N', 'N', one(T), A, B, one(T), C)
end

@inline function _copyto!(::AbstractColumnMajor, dest::AbstractMatrix,
         bc::BMatMatPlusConstMat{T,<:AbstractColumnMajor,<:AbstractColumnMajor}) where T<: BlasFloat
    M,βc = bc.args
    β,C = βc.args
    A,B = M.A, M.B
    _gemm!(dest, 'N', 'N', one(T), A, B, β, C)
end


@inline function _copyto!(::AbstractColumnMajor, dest::AbstractMatrix,
         bc::BConstMatMatPlusMat{T,<:AbstractColumnMajor,<:AbstractColumnMajor}) where T<: BlasFloat
    αM,C = bc.args
    α,M = αM.args
    A,B = M.A, M.B
    _gemm!(dest, 'N', 'N', α, A, B, one(T), C)
end


@inline function _copyto!(::AbstractColumnMajor, dest::AbstractMatrix,
         bc::BConstMatMatPlusConstMat{T,<:AbstractColumnMajor,<:AbstractColumnMajor}) where T<: BlasFloat
    αM,βc = bc.args
    α,M = αM.args
    A,B = M.A, M.B
    β,C = βc.args
    _gemm!(dest, 'N', 'N', α, A, B, β, C)
end

# AdjTrans

@inline function _copyto!(::AbstractColumnMajor, dest::AbstractMatrix,
         bc::BConstMatMatPlusConstMat{T,<:AbstractRowMajor,<:AbstractColumnMajor}) where T<: BlasReal
    αM,βc = bc.args
    α,M = αM.args
    A,B = M.A, M.B
    β,C = βc.args
    _gemm!(dest, 'T', 'N', α, transpose(A), B, β, C)
end

@inline function _copyto!(::AbstractColumnMajor, dest::AbstractMatrix,
         bc::BConstMatMatPlusConstMat{T,<:AbstractRowMajor,<:AbstractColumnMajor}) where T<: BlasComplex
    αM,βc = bc.args
    α,M = αM.args
    A,B = M.A, M.B
    β,C = βc.args
    _gemm!(dest, 'C', 'N', α, A', B, β, C)
end
