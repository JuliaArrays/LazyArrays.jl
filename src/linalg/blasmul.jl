### This support BLAS style multiplication
#           α * A * B + β C
# it represents the same thing as Mul(α, A, B) .+ Mul(β, C)
# but avoids the broadcast machinery

# Lazy representation of α*A*B + β*C
struct BLASMul{StyleA, StyleB, StyleC, T, AA, BB, CC}
    style_A::StyleA
    style_B::StyleB
    style_C::StyleC
    α::T
    A::AA
    B::BB
    β::T
    C::CC
end

@inline BLASMul(α, A, B, β, C) = BLASMul(MemoryLayout(A), MemoryLayout(B), MemoryLayout(C), α, A, B, β, C)

@inline function copyto!(dest::AbstractArray, M::BLASMul)
    M.C ≡ dest || copyto!(dest, M.C)
    materialize!(BLASMul(M.α, M.A, M.B, M.β, dest))
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


# TODO: the blasmul! commands are extraneous
@inline materialize!(M::BLASMul) = blasmul!(M.C, M.A, M.B, M.α, M.β,
                                            M.style_C, M.style_A, M.style_B)

@inline blasmul!(y::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}, α::T, β::T,
              ::AbstractStridedLayout, ::AbstractColumnMajor, ::AbstractStridedLayout) where T<:BlasFloat =
    _gemv!('N', α, A, x, β, y)


@inline blasmul!(y::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}, α::T, β::T,
              ::AbstractStridedLayout, ::AbstractRowMajor, ::AbstractStridedLayout) where T<:BlasFloat =
    _gemv!('T', α, transpose(A), x, β, y)

@inline blasmul!(y::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}, α::T, β::T,
              ::AbstractStridedLayout, ::ConjLayout{<:AbstractRowMajor}, ::AbstractStridedLayout) where T<:BlasComplex =
    _gemv!('C', α, A', x, β, y)



@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::AbstractColumnMajor, ::AbstractColumnMajor, ::AbstractColumnMajor) where T<:BlasFloat =
    _gemm!('N', 'N', α, A, x, β, y)

@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::AbstractColumnMajor, ::AbstractColumnMajor, ::AbstractRowMajor) where T <: BlasFloat =
    _gemm!('N', 'T', α, A, transpose(x), β, y)
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::AbstractColumnMajor, ::AbstractColumnMajor, ::ConjLayout{<:AbstractRowMajor}) where T <: BlasComplex =
    _gemm!('N', 'C', α, A, x', β, y)

@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::AbstractColumnMajor, ::AbstractRowMajor, ::AbstractColumnMajor) where T <: BlasFloat =
    _gemm!('T', 'N', α, transpose(A), x, β, y)
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::AbstractColumnMajor, ::ConjLayout{<:AbstractRowMajor}, ::AbstractColumnMajor) where T <: BlasComplex =
    _gemm!('C', 'N', α, A', x, β, y)


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



@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::AbstractRowMajor, ::AbstractColumnMajor, ::AbstractColumnMajor) where T <: BlasFloat =
    _gemm!('T', 'T', α, x, A, β, transpose(y))
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::ConjLayout{<:AbstractRowMajor}, ::AbstractColumnMajor, ::AbstractColumnMajor) where T <: BlasComplex =
    _gemm!('C', 'C', α, x, A, β, y')

@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::AbstractRowMajor, ::AbstractColumnMajor, ::AbstractRowMajor) where T <: BlasFloat =
    _gemm!('N', 'T', α, transpose(x), A, β, transpose(y))
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::ConjLayout{<:AbstractRowMajor}, ::AbstractColumnMajor, ::AbstractRowMajor) where T <: BlasComplex =
    _gemm!('N', 'T', α, transpose(x), A, β, y')
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::ConjLayout{<:AbstractRowMajor}, ::AbstractColumnMajor, ::ConjLayout{<:AbstractRowMajor}) where T <: BlasComplex =
    _gemm!('N', 'C', α, x', A, β, y')

@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::AbstractRowMajor, ::AbstractRowMajor, ::AbstractColumnMajor) where T <: BlasFloat =
    _gemm!('T', 'N', α, x, transpose(A), β, transpose(y))
@inline blasmul!(y::AbstractMatrix{T}, A::AbstractMatrix{T}, x::AbstractMatrix{T}, α::T, β::T,
              ::ConjLayout{<:AbstractRowMajor}, ::ConjLayout{<:AbstractRowMajor}, ::AbstractColumnMajor) where T <: BlasComplex =
    _gemm!('C', 'N', α, x, A', β, y')

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

@blasmatvec SymmetricLayout{ColumnMajor}
@blasmatvec SymmetricLayout{DenseColumnMajor}

@inline blasmul!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector, α::T, β::T,
              ::AbstractStridedLayout, S::SymmetricLayout{<:AbstractColumnMajor}, ::AbstractStridedLayout) where T<:BlasFloat =
    _symv!(S.uplo, α, symmetricdata(A), x, β, y)

@blasmatvec SymmetricLayout{RowMajor}
@blasmatvec SymmetricLayout{DenseRowMajor}

@inline blasmul!(y::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}, α::T, β::T,
              ::AbstractStridedLayout, S::SymmetricLayout{<:AbstractRowMajor}, ::AbstractStridedLayout) where T<:BlasFloat =
    _symv!(S.uplo == 'L' ? 'U' : 'L', α, transpose(symmetricdata(A)), x, β, y)

@blasmatvec HermitianLayout{ColumnMajor}
@blasmatvec HermitianLayout{DenseColumnMajor}

@inline blasmul!(y::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}, α::T, β::T,
              ::AbstractStridedLayout, S::HermitianLayout{<:AbstractColumnMajor}, ::AbstractStridedLayout) where T<:BlasFloat =
    _hemv!(S.uplo, α, hermitiandata(A), x, β, y)

@blasmatvec HermitianLayout{RowMajor}
@blasmatvec HermitianLayout{DenseRowMajor}

@inline blasmul!(y::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}, α::T, β::T,
              ::AbstractStridedLayout, ::HermitianLayout{<:AbstractRowMajor}, ::AbstractStridedLayout) where T<:BlasComplex =
    _hemv!(S.uplo == 'L' ? 'U' : 'L', α, hermitiandata(A)', x, β, y)


###
# Triangular
###

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatMulVec{<:TriangularLayout{UPLO,UNIT,<:AbstractColumnMajor},
                                   <:AbstractStridedLayout, T, T}) where {UPLO,UNIT,T <: BlasFloat}
    A,x = M.A, M.B
    x ≡ dest || copyto!(dest, x)
    BLAS.trmv!(UPLO, 'N', UNIT, triangulardata(A), dest)
end

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatMulVec{<:TriangularLayout{UPLO,UNIT,<:AbstractRowMajor},
                                   <:AbstractStridedLayout, T, T}) where {UPLO,UNIT,T <: BlasFloat}
    A,x = M.A, M.B
    x ≡ dest || copyto!(dest, x)
    BLAS.trmv!(UPLO, 'T', UNIT, transpose(triangulardata(A)), dest)
end


@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatMulVec{<:TriangularLayout{UPLO,UNIT,<:ConjLayout{<:AbstractRowMajor}},
                                   <:AbstractStridedLayout, T, T}) where {UPLO,UNIT,T <: BlasFloat}
    A,x = M.A, M.B
    x ≡ dest || copyto!(dest, x)
    BLAS.trmv!(UPLO, 'C', UNIT, triangulardata(A)', dest)
end
