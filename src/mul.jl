

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

@inline blasmul!(y, A, x, α, β) = blasmul!(y, A, x, α, β, MemoryLayout(y), MemoryLayout(A), MemoryLayout(x))

@inline function blasmul!(dest, A, x, y, α, β)
    y ≡ dest || copyto!(dest, y)
    blasmul!(dest, A, x, α, β)
end


function _copyto! end

macro blasmatvec(Lay)
    esc(quote
        @inline function LazyArrays._copyto!(::AbstractStridedLayout, dest::AbstractVector,
                 bc::LazyArrays.BMatVec{T, <:$Lay, <:AbstractStridedLayout}) where T<: BlasFloat
            (M,) = bc.args
            A,x = M.A, M.B
            blasmul!(dest, A, x, one(T), zero(T))
        end


        @inline function LazyArrays._copyto!(::LazyArrays.AbstractStridedLayout, dest::AbstractVector{T},
                 bc::LazyArrays.BConstMatVec{T, <:$Lay,<:LazyArrays.AbstractStridedLayout}) where T<: BlasFloat
            α,M = bc.args
            A,x = M.A, M.B
            LazyArrays.blasmul!(dest, A, x, α, zero(T))
        end


        @inline function LazyArrays._copyto!(::LazyArrays.AbstractStridedLayout, dest::AbstractVector,
                 bc::LazyArrays.BMatVecPlusVec{T,<:$Lay,<:LazyArrays.AbstractStridedLayout}) where T<: BlasFloat
            M,y = bc.args
            A,x = M.A, M.B
            LazyArrays.blasmul!(dest, A, x, y, one(T), one(T))
        end

        @inline function LazyArrays._copyto!(::LazyArrays.AbstractStridedLayout, dest::AbstractVector,
                 bc::LazyArrays.BMatVecPlusConstVec{T,<:$Lay,<:LazyArrays.AbstractStridedLayout}) where T<: BlasFloat
            M,βc = bc.args
            β,y = βc.args
            A,x = M.A, M.B
            LazyArrays.blasmul!(dest, A, x, y, one(T), β)
        end


        @inline function LazyArrays._copyto!(::LazyArrays.AbstractStridedLayout, dest::AbstractVector,
                 bc::LazyArrays.BConstMatVecPlusVec{T,<:$Lay,<:LazyArrays.AbstractStridedLayout}) where T<: BlasFloat
            αM,y = bc.args
            α,M = αM.args
            A,x = M.A, M.B
            LazyArrays.blasmul!(dest, A, x, y, α, one(T))
        end


        @inline function LazyArrays._copyto!(::LazyArrays.AbstractStridedLayout, dest::AbstractVector,
                 bc::LazyArrays.BConstMatVecPlusConstVec{T,<:$Lay,<:LazyArrays.AbstractStridedLayout}) where T<: BlasFloat
            αM,βc = bc.args
            α,M = αM.args
            A,x = M.A, M.B
            β,y = βc.args
            LazyArrays.blasmul!(dest, A, x, y, α, β)
        end
    end)
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

macro blasmatmat(CTyp, ATyp, BTyp)
    esc(quote
        @inline function _copyto!(::$CTyp, dest::AbstractMatrix,
                 bc::BMatMat{T,<:$ATyp,<:$BTyp}) where T<: BlasFloat
            (M,) = bc.args
            A,B = M.A, M.B
            blasmul!(dest, A, B, one(T), zero(T))
        end

        @inline function _copyto!(::$CTyp, dest::AbstractMatrix{T},
                 bc::BConstMatMat{T,<:$ATyp,<:$BTyp}) where T<: BlasFloat
            α,M = bc.args
            A,B = M.A, M.B
            blasmul!(dest, A, B, α, zero(T))
        end

        @inline function _copyto!(::$CTyp, dest::AbstractMatrix,
                 bc::BMatMatPlusMat{T,<:$ATyp,<:$BTyp}) where T<: BlasFloat
            M,C = bc.args
            A,B = M.A, M.B
            blasmul!(dest, A, B, C, one(T), one(T))
        end

        @inline function _copyto!(::$CTyp, dest::AbstractMatrix,
                 bc::BMatMatPlusConstMat{T,<:$ATyp,<:$BTyp}) where T<: BlasFloat
            M,βc = bc.args
            β,C = βc.args
            A,B = M.A, M.B
            blasmul!(dest, A, B, C, one(T), β)
        end


        @inline function _copyto!(::$CTyp, dest::AbstractMatrix,
                 bc::BConstMatMatPlusMat{T,<:$ATyp,<:$BTyp}) where T<: BlasFloat
            αM,C = bc.args
            α,M = αM.args
            A,B = M.A, M.B
            blasmul!(dest, A, B, C, α, one(T))
        end


        @inline function _copyto!(::$CTyp, dest::AbstractMatrix,
                 bc::BConstMatMatPlusConstMat{T,<:$ATyp,<:$BTyp}) where T<: BlasFloat
            αM,βc = bc.args
            α,M = αM.args
            A,B = M.A, M.B
            β,C = βc.args
            blasmul!(dest, A, B, C, α, β)
        end
    end)
end


# make copy to make sure always works
@inline function _gemv!(tA, α, A, x, β, y)
    if x ≡ y
        BLAS.gemv!(tA, α, A, copy(x), β, y)
    else
        BLAS.gemv!(tA, α, A, x, β, y)
    end
end

# make copy to make sure always works
@inline function _gemm!(tA, tB, α, A, B, β, C)
    if B ≡ C
        BLAS.gemm!(tA, tB, α, A, copy(B), β, C)
    else
        BLAS.gemm!(tA, tB, α, A, B, β, C)
    end
end

@blasmatvec AbstractColumnMajor

@inline blasmul!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector, α, β,
              ::AbstractStridedLayout, ::AbstractColumnMajor, ::AbstractStridedLayout) =
    _gemv!('N', α, A, x, β, y)

@blasmatvec AbstractRowMajor

@inline blasmul!(y::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}, α, β,
              ::AbstractStridedLayout, ::AbstractRowMajor, ::AbstractStridedLayout) where T<:BlasReal =
    _gemv!('T', α, transpose(A), x, β, y)

@inline blasmul!(y::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}, α, β,
              ::AbstractStridedLayout, ::AbstractRowMajor, ::AbstractStridedLayout) where T<:BlasComplex =
    _gemv!('C', α, A', x, β, y)


@blasmatmat AbstractColumnMajor AbstractColumnMajor AbstractColumnMajor
@inline blasmul!(y::AbstractMatrix, A::AbstractMatrix, x::AbstractMatrix, α, β,
              ::AbstractColumnMajor, ::AbstractColumnMajor, ::AbstractColumnMajor) =
    _gemm!('N', 'N', α, A, x, β, y)

@blasmatmat AbstractColumnMajor AbstractColumnMajor AbstractRowMajor
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α, β,
              ::AbstractColumnMajor, ::AbstractColumnMajor, ::AbstractRowMajor) where T <: BlasReal =
    _gemm!('N', 'T', α, A, transpose(x), β, y)
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α, β,
              ::AbstractColumnMajor, ::AbstractColumnMajor, ::AbstractRowMajor) where T <: BlasComplex =
    _gemm!('N', 'C', α, A, x', β, y)

@blasmatmat AbstractColumnMajor AbstractRowMajor AbstractColumnMajor
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α, β,
              ::AbstractColumnMajor, ::AbstractRowMajor, ::AbstractColumnMajor) where T <: BlasReal =
    _gemm!('T', 'N', α, transpose(A), x, β, y)
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α, β,
              ::AbstractColumnMajor, ::AbstractRowMajor, ::AbstractColumnMajor) where T <: BlasComplex =
    _gemm!('C', 'N', α, A', x, β, y)

@blasmatmat AbstractColumnMajor AbstractRowMajor AbstractRowMajor
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α, β,
              ::AbstractColumnMajor, ::AbstractRowMajor, ::AbstractRowMajor) where T <: BlasReal =
    _gemm!('T', 'T', α, transpose(A), transpose(x), β, y)
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α, β,
              ::AbstractColumnMajor, ::AbstractRowMajor, ::AbstractRowMajor) where T <: BlasComplex =
    _gemm!('C', 'C', α, A', x', β, y)


@blasmatmat AbstractRowMajor AbstractColumnMajor AbstractColumnMajor
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α, β,
              ::AbstractRowMajor, ::AbstractColumnMajor, ::AbstractColumnMajor) where T <: BlasReal =
    _gemm!('T', 'T', α, x, A, β, transpose(y))
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α, β,
              ::AbstractRowMajor, ::AbstractColumnMajor, ::AbstractColumnMajor) where T <: BlasComplex =
    _gemm!('C', 'C', α, x, A, β, y')

@blasmatmat AbstractRowMajor AbstractColumnMajor AbstractRowMajor
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α, β,
              ::AbstractRowMajor, ::AbstractColumnMajor, ::AbstractRowMajor) where T <: BlasReal =
    _gemm!('N', 'T', α, transpose(x), A, β, transpose(y))
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α, β,
              ::AbstractRowMajor, ::AbstractColumnMajor, ::AbstractRowMajor) where T <: BlasComplex =
    _gemm!('N', 'C', α, x', A, β, y')

@blasmatmat AbstractRowMajor AbstractRowMajor AbstractColumnMajor
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α, β,
              ::AbstractRowMajor, ::AbstractRowMajor, ::AbstractColumnMajor) where T <: BlasReal =
    _gemm!('T', 'N', α, x, transpose(A), β, transpose(y))
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α, β,
              ::AbstractRowMajor, ::AbstractRowMajor, ::AbstractColumnMajor) where T <: BlasComplex =
    _gemm!('C', 'N', α, x, A', β, y')

@blasmatmat AbstractRowMajor AbstractRowMajor AbstractRowMajor
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α, β,
              ::AbstractRowMajor, ::AbstractRowMajor, ::AbstractRowMajor) where T <: BlasReal =
    _gemm!('N', 'N', α, transpose(x), transpose(A), β, transpose(y))
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α, β,
              ::AbstractRowMajor, ::AbstractRowMajor, ::AbstractRowMajor) where T <: BlasComplex =
    _gemm!('N', 'N', α, x', A', β, y')
