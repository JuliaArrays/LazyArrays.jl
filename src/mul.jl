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
const ArrayMulArray{TV, styleA, styleB, p, q, T, V} =
    Mul{TV, styleA, styleB, <:AbstractArray{T,p}, <:AbstractArray{V,q}}

const BArrayMulArray{TV, styleA, styleB, p, q, T, V} =
    Broadcasted{ArrayMulArrayStyle{styleA,styleB,p,q}, <:Any, typeof(identity),
                <:Tuple{<:ArrayMulArray{TV,styleA,styleB,p,q,T,V}}}
const BConstArrayMulArray{T, styleA, styleB, p, q} =
    Broadcasted{ArrayMulArrayStyle{styleA,styleB,p,q},
                    <:Any, typeof(*),
                    <:Tuple{T,<:ArrayMulArray{T,styleA,styleB,p,q,T,T}}}
const BArrayMulArrayPlusArray{T, styleA, styleB, p, q} =
    Broadcasted{ArrayMulArrayStyle{styleA,styleB,p,q},
                <:Any, typeof(+),
                <:Tuple{<:ArrayMulArray{T,styleA,styleB,p,q,T,T},<:AbstractArray{T,q}}}
const BArrayMulArrayPlusConstArray{T, styleA, styleB, p, q} =
    Broadcasted{ArrayMulArrayStyle{styleA,styleB,p,q},
                <:Any, typeof(+),
                <:Tuple{<:ArrayMulArray{T,styleA,styleB,p,q,T,T},
                Broadcasted{DefaultArrayStyle{q},<:Any,typeof(*),
                            <:Tuple{T,<:AbstractArray{T,q}}}}}
const BConstArrayMulArrayPlusArray{T, styleA, styleB, p, q} =
    Broadcasted{ArrayMulArrayStyle{styleA,styleB,p,q},
                <:Any, typeof(+),
                <:Tuple{Broadcasted{<:ArrayMulArrayStyle{styleA,styleB,p,q},
                                    <:Any, typeof(*),
                                    <:Tuple{T,<:ArrayMulArray{T,styleA,styleB,p,q,T,T}}},
                        <:AbstractArray{T,q}}}
const BConstArrayMulArrayPlusConstArray{T, styleA, styleB, p, q} =
    Broadcasted{ArrayMulArrayStyle{styleA,styleB,p,q},
                <:Any, typeof(+),
                <:Tuple{Broadcasted{<:ArrayMulArrayStyle{styleA,styleB,p,q},
                                    <:Any, typeof(*),
                                    <:Tuple{T,<:ArrayMulArray{T,styleA,styleB,p,q,T,T}}},
                        Broadcasted{DefaultArrayStyle{q},<:Any,typeof(*),<:Tuple{T,<:AbstractArray{T,q}}}}}


BroadcastStyle(::Type{<:ArrayMulArray{<:Any,StyleA,StyleB,p,q}}) where {StyleA,StyleB,p,q} = ArrayMulArrayStyle{StyleA,StyleB,p,q}()
BroadcastStyle(M::ArrayMulArrayStyle, ::DefaultArrayStyle) = M
BroadcastStyle(::DefaultArrayStyle, M::ArrayMulArrayStyle) = M
similar(M::Broadcasted{<:ArrayMulArrayStyle}, ::Type{ElType}) where ElType = Array{Eltype}(undef,size(M.args[1]))

@inline copyto!(dest::AbstractArray, M::Mul) = _copyto!(MemoryLayout(dest), dest, M)
# default to Base mul!
function _copyto!(_, dest::AbstractArray, M::ArrayMulArray)
    A,x = M.A, M.B
    mul!(dest, A, x)
end

@inline copyto!(dest::AbstractArray, bc::Broadcasted{<:ArrayMulArrayStyle}) =
    _copyto!(MemoryLayout(dest), dest, bc)
# Use default broacasting in general
@inline _copyto!(_, dest, bc::Broadcasted) = copyto!(dest, Broadcasted{Nothing}(bc.f, bc.args, bc.axes))

# Use copyto! for y .= Mul(A,b)
@inline function _copyto!(_, dest::AbstractArray, bc::BArrayMulArray)
    (M,) = bc.args
    copyto!(dest, M)
end


####
# Matrix * Vector
####
let (p,q) = (2,1)
    global const MatMulVecStyle{StyleA, StyleB} = ArrayMulArrayStyle{StyleA, StyleB, p, q}
    global const MatMulVec{TV, styleA, styleB, T, V} = ArrayMulArray{TV, styleA, styleB, p, q, T, V}

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

# Matrix * Vector

# @inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
#          bc::BMatVec{T, <:AbstractColumnMajor, <:AbstractStridedLayout}) where T<: BlasFloat
#     (M,) = bc.args
#     dest .= one(T) .* M .+ zero(T) .* dest
# end

# support mul! by calling lazy mul
macro lazymul(Typ)
    esc(quote
        LinearAlgebra.mul!(dest::AbstractVector, A::$Typ, x::AbstractVector) =
            (dest .= Mul(A,x))

        LinearAlgebra.mul!(dest::AbstractMatrix, A::$Typ, x::AbstractMatrix) =
            (dest .= Mul(A,x))
        LinearAlgebra.mul!(dest::AbstractMatrix, A::$Typ, x::$Typ) =
            (dest .= Mul(A,x))
        LinearAlgebra.mul!(dest::AbstractMatrix, A::$Typ, x::Adjoint{<:Any,<:AbstractMatrix}) =
            (dest .= Mul(A,x))

        LinearAlgebra.mul!(dest::AbstractVector, A::Adjoint{<:Any,<:$Typ}, b::AbstractVector) =
            (dest .= Mul(A, b))
        LinearAlgebra.mul!(dest::AbstractVector, A::Transpose{<:Any,<:$Typ}, b::AbstractVector) =
            (dest .= Mul(A, b))

        LinearAlgebra.mul!(dest::AbstractVector, A::Symmetric{<:Any,<:$Typ}, b::AbstractVector) =
            (dest .= Mul(A, b))
        LinearAlgebra.mul!(dest::AbstractVector, A::Hermitian{<:Any,<:$Typ}, b::AbstractVector) =
            (dest .= Mul(A, b))
    end)
end

macro lazylmul(Typ)
    esc(quote
        LinearAlgebra.lmul!(A::$Typ, x::AbstractVector) = (x .= Mul(A,x))
        LinearAlgebra.lmul!(A::$Typ, x::AbstractMatrix) = (x .= Mul(A,x))
        LinearAlgebra.lmul!(A::$Typ, x::StridedVector) = (x .= Mul(A,x))
        LinearAlgebra.lmul!(A::$Typ, x::StridedMatrix) = (x .= Mul(A,x))
    end)
end

@inline blasmul!(y, A, x, α, β) = blasmul!(y, A, x, α, β, MemoryLayout(y), MemoryLayout(A), MemoryLayout(x))

@inline function blasmul!(dest, A, x, y, α, β)
    y ≡ dest || copyto!(dest, y)
    blasmul!(dest, A, x, α, β)
end

macro _blasmatvec(Lay, Typ)
    esc(quote
        # y .= Mul(A,b) gets lowered here
        @inline function LazyArrays._copyto!(::LazyArrays.AbstractStridedLayout, dest::AbstractVector{T},
                                             M::LazyArrays.MatMulVec{T, <:$Lay, <:LazyArrays.AbstractStridedLayout, T, T}) where T<: $Typ
            A,x = M.A, M.B
            LazyArrays.blasmul!(dest, A, x, one(T), zero(T))
        end

        @inline function LazyArrays._copyto!(::LazyArrays.AbstractStridedLayout, dest::AbstractVector{T},
                 bc::LazyArrays.BConstMatVec{T, <:$Lay,<:LazyArrays.AbstractStridedLayout}) where T<: $Typ
            α,M = bc.args
            A,x = M.A, M.B
            LazyArrays.blasmul!(dest, A, x, α, zero(T))
        end

        @inline function LazyArrays._copyto!(::LazyArrays.AbstractStridedLayout, dest::AbstractVector{T},
                 bc::LazyArrays.BMatVecPlusVec{T,<:$Lay,<:LazyArrays.AbstractStridedLayout}) where T<: $Typ
            M,y = bc.args
            A,x = M.A, M.B
            LazyArrays.blasmul!(dest, A, x, y, one(T), one(T))
        end

        @inline function LazyArrays._copyto!(::LazyArrays.AbstractStridedLayout, dest::AbstractVector{T},
                 bc::LazyArrays.BMatVecPlusConstVec{T,<:$Lay,<:LazyArrays.AbstractStridedLayout}) where T<: $Typ
            M,βc = bc.args
            β,y = βc.args
            A,x = M.A, M.B
            LazyArrays.blasmul!(dest, A, x, y, one(T), β)
        end

        @inline function LazyArrays._copyto!(::LazyArrays.AbstractStridedLayout, dest::AbstractVector{T},
                 bc::LazyArrays.BConstMatVecPlusVec{T,<:$Lay,<:LazyArrays.AbstractStridedLayout}) where T<: $Typ
            αM,y = bc.args
            α,M = αM.args
            A,x = M.A, M.B
            LazyArrays.blasmul!(dest, A, x, y, α, one(T))
        end

        @inline function LazyArrays._copyto!(::LazyArrays.AbstractStridedLayout, dest::AbstractVector{T},
                 bc::LazyArrays.BConstMatVecPlusConstVec{T,<:$Lay,<:LazyArrays.AbstractStridedLayout}) where T<: $Typ
            αM,βc = bc.args
            α,M = αM.args
            A,x = M.A, M.B
            β,y = βc.args
            LazyArrays.blasmul!(dest, A, x, y, α, β)
        end
    end)
end

macro blasmatvec(Lay)
    esc(quote
        LazyArrays.@_blasmatvec $Lay LinearAlgebra.BLAS.BlasFloat
        LazyArrays.@_blasmatvec LazyArrays.ConjLayout{<:$Lay} LinearAlgebra.BLAS.BlasComplex
    end)
end



####
# Matrix * Matrix
####


let (p,q) = (2,2)
    global const MatMulMatStyle{StyleA, StyleB} = ArrayMulArrayStyle{StyleA, StyleB, p, q}
    global const MatMulMat{TV, styleA, styleB, T, V} = ArrayMulArray{TV, styleA, styleB, p, q, T, V}

    global const BMatMat{TV, styleA, styleB, T, V} = BArrayMulArray{TV, styleA, styleB, p, q, T, V}
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

# getindex(M::MatMulMat, kj::CartesianIndex{2}) = M[kj...]


# Matrix * Vector

# this should work but for some reason was not inlining correctly
# @inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
#          bc::BMatVec{T, <:AbstractColumnMajor, <:AbstractStridedLayout}) where T<: BlasFloat
#     (M,) = bc.args
#     dest .= one(T) .* M .+ zero(T) .* dest
# end

macro _blasmatmat(CTyp, ATyp, BTyp, Typ)
    esc(quote
        @inline function LazyArrays._copyto!(::$CTyp, dest::AbstractMatrix{T},
                 M::LazyArrays.MatMulMat{T,<:$ATyp,<:$BTyp,T,T}) where T<: $Typ
            A,B = M.A, M.B
            LazyArrays.blasmul!(dest, A, B, one(T), zero(T))
        end

        @inline function LazyArrays._copyto!(::$CTyp, dest::AbstractMatrix{T},
                 bc::LazyArrays.BConstMatMat{T,<:$ATyp,<:$BTyp}) where T<: $Typ
            α,M = bc.args
            A,B = M.A, M.B
            LazyArrays.blasmul!(dest, A, B, α, zero(T))
        end

        @inline function LazyArrays._copyto!(::$CTyp, dest::AbstractMatrix{T},
                 bc::LazyArrays.BMatMatPlusMat{T,<:$ATyp,<:$BTyp}) where T<: $Typ
            M,C = bc.args
            A,B = M.A, M.B
            LazyArrays.blasmul!(dest, A, B, C, one(T), one(T))
        end

        @inline function LazyArrays._copyto!(::$CTyp, dest::AbstractMatrix{T},
                 bc::LazyArrays.BMatMatPlusConstMat{T,<:$ATyp,<:$BTyp}) where T<: $Typ
            M,βc = bc.args
            β,C = βc.args
            A,B = M.A, M.B
            LazyArrays.blasmul!(dest, A, B, C, one(T), β)
        end

        @inline function LazyArrays._copyto!(::$CTyp, dest::AbstractMatrix{T},
                 bc::LazyArrays.BConstMatMatPlusMat{T,<:$ATyp,<:$BTyp}) where T<: $Typ
            αM,C = bc.args
            α,M = αM.args
            A,B = M.A, M.B
            LazyArrays.blasmul!(dest, A, B, C, α, one(T))
        end

        @inline function LazyArrays._copyto!(::$CTyp, dest::AbstractMatrix{T},
                 bc::LazyArrays.BConstMatMatPlusConstMat{T,<:$ATyp,<:$BTyp}) where T<: $Typ
            αM,βc = bc.args
            α,M = αM.args
            A,B = M.A, M.B
            β,C = βc.args
            LazyArrays.blasmul!(dest, A, B, C, α, β)
        end
    end)
end


macro blasmatmat(ATyp, BTyp, CTyp)
    esc(quote
        LazyArrays.@_blasmatmat $ATyp $BTyp $CTyp BlasFloat
        LazyArrays.@_blasmatmat LazyArrays.ConjLayout{<:$ATyp} $BTyp $CTyp BlasComplex
        LazyArrays.@_blasmatmat $ATyp LazyArrays.ConjLayout{<:$BTyp} $CTyp BlasComplex
        LazyArrays.@_blasmatmat LazyArrays.ConjLayout{<:$ATyp} LazyArrays.ConjLayout{<:$BTyp} $CTyp BlasComplex
        LazyArrays.@_blasmatmat $ATyp $BTyp LazyArrays.ConjLayout{<:$CTyp} BlasComplex
        LazyArrays.@_blasmatmat LazyArrays.ConjLayout{<:$ATyp} $BTyp LazyArrays.ConjLayout{<:$CTyp} BlasComplex
        LazyArrays.@_blasmatmat $ATyp LazyArrays.ConjLayout{<:$BTyp} LazyArrays.ConjLayout{<:$CTyp} BlasComplex
        LazyArrays.@_blasmatmat LazyArrays.ConjLayout{<:$ATyp} LazyArrays.ConjLayout{<:$BTyp} LazyArrays.ConjLayout{<:$CTyp} BlasComplex
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

@inline blasmul!(y::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}, α::T, β::T,
              ::AbstractStridedLayout, ::AbstractColumnMajor, ::AbstractStridedLayout) where T<:BlasFloat =
    _gemv!('N', α, A, x, β, y)

@blasmatvec AbstractRowMajor

@inline blasmul!(y::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}, α::T, β::T,
              ::AbstractStridedLayout, ::AbstractRowMajor, ::AbstractStridedLayout) where T<:BlasFloat =
    _gemv!('T', α, transpose(A), x, β, y)

@inline blasmul!(y::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}, α::T, β::T,
              ::AbstractStridedLayout, ::ConjLayout{<:AbstractRowMajor}, ::AbstractStridedLayout) where T<:BlasComplex =
    _gemv!('C', α, A', x, β, y)


@blasmatmat AbstractColumnMajor AbstractColumnMajor AbstractColumnMajor
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::AbstractColumnMajor, ::AbstractColumnMajor, ::AbstractColumnMajor) where T<:BlasFloat =
    _gemm!('N', 'N', α, A, x, β, y)

@blasmatmat AbstractColumnMajor AbstractColumnMajor AbstractRowMajor
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::AbstractColumnMajor, ::AbstractColumnMajor, ::AbstractRowMajor) where T <: BlasFloat =
    _gemm!('N', 'T', α, A, transpose(x), β, y)
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::AbstractColumnMajor, ::AbstractColumnMajor, ::ConjLayout{<:AbstractRowMajor}) where T <: BlasComplex =
    _gemm!('N', 'C', α, A, x', β, y)

@blasmatmat AbstractColumnMajor AbstractRowMajor AbstractColumnMajor
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::AbstractColumnMajor, ::AbstractRowMajor, ::AbstractColumnMajor) where T <: BlasFloat =
    _gemm!('T', 'N', α, transpose(A), x, β, y)
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::AbstractColumnMajor, ::ConjLayout{<:AbstractRowMajor}, ::AbstractColumnMajor) where T <: BlasComplex =
    _gemm!('C', 'N', α, A', x, β, y)

@blasmatmat AbstractColumnMajor AbstractRowMajor AbstractRowMajor
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::AbstractColumnMajor, ::AbstractRowMajor, ::AbstractRowMajor) where T <: BlasFloat =
    _gemm!('T', 'T', α, transpose(A), transpose(x), β, y)
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::AbstractColumnMajor, ::AbstractRowMajor, ::ConjLayout{<:AbstractRowMajor}) where T <: BlasComplex =
    _gemm!('T', 'C', α, transpose(A), x', β, y)
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::AbstractColumnMajor, ::ConjLayout{<:AbstractRowMajor}, ::AbstractRowMajor) where T <: BlasComplex =
    _gemm!('C', 'T', α, A', x', β, y)
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::AbstractColumnMajor, ::ConjLayout{<:AbstractRowMajor}, ::ConjLayout{<:AbstractRowMajor}) where T <: BlasComplex =
    _gemm!('C', 'C', α, A', x', β, y)


@blasmatmat AbstractRowMajor AbstractColumnMajor AbstractColumnMajor
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::AbstractRowMajor, ::AbstractColumnMajor, ::AbstractColumnMajor) where T <: BlasFloat =
    _gemm!('T', 'T', α, x, A, β, transpose(y))
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::ConjLayout{<:AbstractRowMajor}, ::AbstractColumnMajor, ::AbstractColumnMajor) where T <: BlasComplex =
    _gemm!('C', 'C', α, x, A, β, y')

@blasmatmat AbstractRowMajor AbstractColumnMajor AbstractRowMajor
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::AbstractRowMajor, ::AbstractColumnMajor, ::AbstractRowMajor) where T <: BlasFloat =
    _gemm!('N', 'T', α, transpose(x), A, β, transpose(y))
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::ConjLayout{<:AbstractRowMajor}, ::AbstractColumnMajor, ::AbstractRowMajor) where T <: BlasComplex =
    _gemm!('N', 'T', α, transpose(x), A, β, y')
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::ConjLayout{<:AbstractRowMajor}, ::AbstractColumnMajor, ::ConjLayout{<:AbstractRowMajor}) where T <: BlasComplex =
    _gemm!('N', 'C', α, x', A, β, y')

@blasmatmat AbstractRowMajor AbstractRowMajor AbstractColumnMajor
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::AbstractRowMajor, ::AbstractRowMajor, ::AbstractColumnMajor) where T <: BlasFloat =
    _gemm!('T', 'N', α, x, transpose(A), β, transpose(y))
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::ConjLayout{<:AbstractRowMajor}, ::ConjLayout{<:AbstractRowMajor}, ::AbstractColumnMajor) where T <: BlasComplex =
    _gemm!('C', 'N', α, x, A', β, y')

@blasmatmat AbstractRowMajor AbstractRowMajor AbstractRowMajor
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::AbstractRowMajor, ::AbstractRowMajor, ::AbstractRowMajor) where T <: BlasFloat =
    _gemm!('N', 'N', α, transpose(x), transpose(A), β, transpose(y))
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::ConjLayout{<:AbstractRowMajor}, ::ConjLayout{<:AbstractRowMajor}, ::ConjLayout{<:AbstractRowMajor}) where T <: BlasComplex =
    _gemm!('N', 'N', α, x', A', β, y')


###
# Symmetric
###

# make copy to make sure always works
@inline function _symv!(tA, α, A, x, β, y)
    if x ≡ y
        BLAS.symv!(tA, α, A, copy(x), β, y)
    else
        BLAS.symv!(tA, α, A, x, β, y)
    end
end

@inline function _hemv!(tA, α, A, x, β, y)
    if x ≡ y
        BLAS.hemv!(tA, α, A, copy(x), β, y)
    else
        BLAS.hemv!(tA, α, A, x, β, y)
    end
end

@blasmatvec SymmetricLayout{<:AbstractColumnMajor}

@inline blasmul!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector, α::T, β::T,
              ::AbstractStridedLayout, S::SymmetricLayout{<:AbstractColumnMajor}, ::AbstractStridedLayout) where T<:BlasFloat =
    _symv!(S.uplo, α, symmetricdata(A), x, β, y)

@blasmatvec SymmetricLayout{<:AbstractRowMajor}

@inline blasmul!(y::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}, α::T, β::T,
              ::AbstractStridedLayout, S::SymmetricLayout{<:AbstractRowMajor}, ::AbstractStridedLayout) where T<:BlasFloat =
    _symv!(S.uplo == 'L' ? 'U' : 'L', α, transpose(symmetricdata(A)), x, β, y)

@blasmatvec HermitianLayout{<:AbstractColumnMajor}

@inline blasmul!(y::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}, α::T, β::T,
              ::AbstractStridedLayout, S::HermitianLayout{<:AbstractColumnMajor}, ::AbstractStridedLayout) where T<:BlasFloat =
    _hemv!(S.uplo, α, hermitiandata(A), x, β, y)

@blasmatvec HermitianLayout{<:AbstractRowMajor}

@inline blasmul!(y::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}, α::T, β::T,
              ::AbstractStridedLayout, ::HermitianLayout{<:AbstractRowMajor}, ::AbstractStridedLayout) where T<:BlasComplex =
    _hemv!(S.uplo == 'L' ? 'U' : 'L', α, hermitiandata(A)', x, β, y)


###
# Triangular
###

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatMulVec{T, <:TriangularLayout{UPLO,UNIT,<:AbstractColumnMajor},
                                   <:AbstractStridedLayout, T, T}) where {UPLO,UNIT,T <: BlasFloat}
    A,x = M.A, M.B
    x ≡ dest || copyto!(dest, x)
    BLAS.trmv!(UPLO, 'N', UNIT, triangulardata(A), dest)
end

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatMulVec{T, <:TriangularLayout{UPLO,UNIT,<:AbstractRowMajor},
                                   <:AbstractStridedLayout, T, T}) where {UPLO,UNIT,T <: BlasFloat}
    A,x = M.A, M.B
    x ≡ dest || copyto!(dest, x)
    BLAS.trmv!(UPLO, 'T', UNIT, transpose(triangulardata(A)), dest)
end


@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatMulVec{T, <:TriangularLayout{UPLO,UNIT,<:ConjLayout{<:AbstractRowMajor}},
                                   <:AbstractStridedLayout, T, T}) where {UPLO,UNIT,T <: BlasFloat}
    A,x = M.A, M.B
    x ≡ dest || copyto!(dest, x)
    BLAS.trmv!(UPLO, 'C', UNIT, triangulardata(A)', dest)
end
