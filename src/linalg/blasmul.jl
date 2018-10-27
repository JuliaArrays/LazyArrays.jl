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

const BLASMatMulVec{StyleA,StyleB,StyleC,T} = BLASMul{StyleA,StyleB,StyleC,T,<:AbstractMatrix{T},<:AbstractVector{T},<:AbstractVector{T}}
const BLASMatMulMat{StyleA,StyleB,StyleC,T} = BLASMul{StyleA,StyleB,StyleC,T,<:AbstractMatrix{T},<:AbstractMatrix{T},<:AbstractMatrix{T}}

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


@inline materialize!(M::BLASMatMulVec{<:AbstractColumnMajor,<:AbstractStridedLayout,<:AbstractStridedLayout,<:BlasFloat}) =
    _gemv!('N', M.α, M.A, M.B, M.β, M.C)
@inline materialize!(M::BLASMatMulVec{<:AbstractRowMajor,<:AbstractStridedLayout,<:AbstractStridedLayout,<:BlasFloat}) =
    _gemv!('T', M.α, transpose(M.A), M.B, M.β, M.C)
@inline materialize!(M::BLASMatMulVec{<:ConjLayout{<:AbstractRowMajor},<:AbstractStridedLayout,<:AbstractStridedLayout,<:BlasComplex}) =
    _gemv!('C', M.α, M.A', M.B, M.β, M.C)

@inline materialize!(M::BLASMatMulMat{<:AbstractColumnMajor,<:AbstractColumnMajor,<:AbstractColumnMajor,<:BlasFloat}) =
    _gemm!('N', 'N', M.α, M.A, M.B, M.β, M.C)
@inline materialize!(M::BLASMatMulMat{<:AbstractColumnMajor,<:AbstractRowMajor,<:AbstractColumnMajor,<:BlasFloat}) =
    _gemm!('N', 'T', M.α, M.A, transpose(M.B), M.β, M.C)
@inline materialize!(M::BLASMatMulMat{<:AbstractColumnMajor,<:ConjLayout{<:AbstractRowMajor},<:AbstractColumnMajor,<:BlasComplex}) =
    _gemm!('N', 'C', M.α, M.A, M.B', M.β, M.C)

@inline materialize!(M::BLASMatMulMat{<:AbstractRowMajor,<:AbstractColumnMajor,<:AbstractColumnMajor,<:BlasFloat}) =
    _gemm!('T', 'N', M.α, transpose(M.A), M.B, M.β, M.C)
@inline materialize!(M::BLASMatMulMat{<:ConjLayout{<:AbstractRowMajor},<:AbstractColumnMajor,<:AbstractColumnMajor,<:BlasComplex}) =
    _gemm!('C', 'N', M.α, M.A', M.B, M.β, M.C)

@inline materialize!(M::BLASMatMulMat{<:AbstractRowMajor,<:AbstractRowMajor,<:AbstractColumnMajor,<:BlasFloat}) =
    _gemm!('T', 'T', M.α, transpose(M.A), transpose(M.B), M.β, M.C)
@inline materialize!(M::BLASMatMulMat{<:AbstractRowMajor,<:ConjLayout{<:AbstractRowMajor},<:AbstractColumnMajor,<:BlasComplex}) =
    _gemm!('T', 'C', M.α, transpose(M.A), M.B', M.β, M.C)

@inline materialize!(M::BLASMatMulMat{<:ConjLayout{<:AbstractRowMajor},<:AbstractRowMajor,<:AbstractColumnMajor,<:BlasComplex}) =
    _gemm!('C', 'T', M.α, M.A', M.B', M.β, M.C)
@inline materialize!(M::BLASMatMulMat{<:ConjLayout{<:AbstractRowMajor},<:ConjLayout{<:AbstractRowMajor},<:AbstractColumnMajor,<:BlasComplex}) =
    _gemm!('C', 'C', M.α, M.A', M.B', M.β, M.C)

@inline materialize!(M::BLASMatMulMat{<:AbstractColumnMajor,<:AbstractColumnMajor,<:AbstractRowMajor,<:BlasFloat}) =
    _gemm!('T', 'T', M.α, M.B, M.A, M.β, transpose(M.C))
@inline materialize!(M::BLASMatMulMat{<:AbstractColumnMajor,<:AbstractColumnMajor,<:ConjLayout{<:AbstractRowMajor},<:BlasComplex}) =
    _gemm!('C', 'C', M.α, M.B, M.A, M.β, M.C')

@inline materialize!(M::BLASMatMulMat{<:AbstractColumnMajor,<:AbstractRowMajor,<:AbstractRowMajor,<:BlasFloat}) =
    _gemm!('N', 'T', M.α, transpose(M.B), M.A, M.β, transpose(M.C))
@inline materialize!(M::BLASMatMulMat{<:AbstractColumnMajor,<:AbstractRowMajor,<:ConjLayout{<:AbstractRowMajor},<:BlasComplex}) =
    _gemm!('N', 'T', M.α, transpose(M.B), M.A, M.β, M.C')
@inline materialize!(M::BLASMatMulMat{<:AbstractColumnMajor,<:ConjLayout{<:AbstractRowMajor},<:ConjLayout{<:AbstractRowMajor},<:BlasComplex}) =
    _gemm!('N', 'C', M.α, M.B', M.A, M.β, M.C')

@inline materialize!(M::BLASMatMulMat{<:AbstractRowMajor,<:AbstractColumnMajor,<:AbstractRowMajor,<:BlasFloat}) =
    _gemm!('T', 'N', M.α, M.B, transpose(M.A), M.β, transpose(M.C))
@inline materialize!(M::BLASMatMulMat{<:ConjLayout{<:AbstractRowMajor},<:AbstractColumnMajor,<:ConjLayout{<:AbstractRowMajor},<:BlasComplex}) =
    _gemm!('C', 'N', M.α, M.B, M.A', M.β, M.C')


@inline materialize!(M::BLASMatMulMat{<:AbstractRowMajor,<:AbstractRowMajor,<:AbstractRowMajor,<:BlasFloat}) =
    _gemm!('N', 'N', M.α, transpose(M.B), transpose(M.A), M.β, transpose(M.C))
@inline materialize!(M::BLASMatMulMat{<:ConjLayout{<:AbstractRowMajor},<:ConjLayout{<:AbstractRowMajor},<:ConjLayout{<:AbstractRowMajor},<:BlasComplex}) =
    _gemm!('N', 'N', M.α, M.B', M.A', M.β, M.C')


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


materialize!(M::BLASMatMulVec{<:SymmetricLayout{<:AbstractColumnMajor},<:AbstractStridedLayout,<:AbstractStridedLayout,<:BlasFloat}) =
    _symv!(M.style_A.uplo, M.α, symmetricdata(M.A), M.B, M.β, M.C)

@blasmatvec SymmetricLayout{RowMajor}
@blasmatvec SymmetricLayout{DenseRowMajor}

materialize!(M::BLASMatMulVec{<:SymmetricLayout{<:AbstractRowMajor},<:AbstractStridedLayout,<:AbstractStridedLayout,<:BlasFloat}) =
    _symv!(M.style_A.uplo == 'L' ? 'U' : 'L', M.α, transpose(symmetricdata(M.A)), M.B, M.β, M.C)

@blasmatvec HermitianLayout{ColumnMajor}
@blasmatvec HermitianLayout{DenseColumnMajor}

materialize!(M::BLASMatMulVec{<:HermitianLayout{<:AbstractColumnMajor},<:AbstractStridedLayout,<:AbstractStridedLayout,<:BlasComplex}) =
    _hemv!(M.style_A.uplo, M.α, hermitiandata(M.A), M.B, M.β, M.C)

@blasmatvec HermitianLayout{RowMajor}
@blasmatvec HermitianLayout{DenseRowMajor}

materialize!(M::BLASMatMulVec{<:HermitianLayout{<:AbstractRowMajor},<:AbstractStridedLayout,<:AbstractStridedLayout,<:BlasComplex}) =
    _hemv!(M.style_A.uplo == 'L' ? 'U' : 'L', M.α, hermitiandata(M.A)', M.B, M.β, M.C)


###
# Triangular
###

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatMulVec{<:TriangularLayout{UPLO,UNIT,<:AbstractColumnMajor},
                                   <:AbstractStridedLayout, T, T}) where {UPLO,UNIT,T <: BlasFloat}
    A,x = M.factors
    x ≡ dest || copyto!(dest, x)
    BLAS.trmv!(UPLO, 'N', UNIT, triangulardata(A), dest)
end

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatMulVec{<:TriangularLayout{UPLO,UNIT,<:AbstractRowMajor},
                                   <:AbstractStridedLayout, T, T}) where {UPLO,UNIT,T <: BlasFloat}
    A,x = M.factors
    x ≡ dest || copyto!(dest, x)
    BLAS.trmv!(UPLO, 'T', UNIT, transpose(triangulardata(A)), dest)
end


@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatMulVec{<:TriangularLayout{UPLO,UNIT,<:ConjLayout{<:AbstractRowMajor}},
                                   <:AbstractStridedLayout, T, T}) where {UPLO,UNIT,T <: BlasFloat}
    A,x = M.factors
    x ≡ dest || copyto!(dest, x)
    BLAS.trmv!(UPLO, 'C', UNIT, triangulardata(A)', dest)
end
