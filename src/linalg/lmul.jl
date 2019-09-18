struct Lmul{StyleA, StyleB, TypeA, TypeB}
    A::TypeA
    B::TypeB
end

Lmul(A::TypeA, B::TypeB) where {TypeA,TypeB} = Lmul{typeof(MemoryLayout(TypeA)),typeof(MemoryLayout(TypeB)),TypeA,TypeB}(A,B)

BroadcastStyle(::Type{<:Lmul}) = ApplyBroadcastStyle()


struct LmulStyle <: AbstractArrayApplyStyle end
# combine_mul_styles(::DiagonalLayout) = LmulStyle()

broadcastable(M::Lmul) = M

const MatLmulVec{StyleA,StyleB} = Lmul{StyleA,StyleB,<:AbstractMatrix,<:AbstractVector}
const MatLmulMat{StyleA,StyleB} = Lmul{StyleA,StyleB,<:AbstractMatrix,<:AbstractMatrix}

const BlasMatLmulVec{StyleA,StyleB,T<:BlasFloat} = Lmul{StyleA,StyleB,<:AbstractMatrix{T},<:AbstractVector{T}}
const BlasMatLmulMat{StyleA,StyleB,T<:BlasFloat} = Lmul{StyleA,StyleB,<:AbstractMatrix{T},<:AbstractMatrix{T}}

####
# LMul materialize
####

copy(M::Mul{LmulStyle}) = copyto!(similar(M), Lmul(M.args...))

@inline copyto!(dest::AbstractVecOrMat, M::Mul{LmulStyle}) = copyto!(dest, Lmul(M.args...))

@inline function materialize!(M::Mul{LmulStyle})
    A,x = M.args
    materialize!(Lmul(A, x))
end

copy(M::Lmul) = materialize!(Lmul(M.A,copy(M.B)))
@inline function copyto!(dest::AbstractArray, M::Lmul)
    M.B ≡ dest || copyto!(dest, M.B)
    materialize!(Lmul(M.A,dest))
end

materialize!(M::Lmul) = lmul!(M.A,M.B)



###
# Triangular
###
mulapplystyle(::TriangularLayout, ::AbstractStridedLayout) = LmulStyle()




@inline function materialize!(M::BlasMatLmulVec{<:TriangularLayout{UPLO,UNIT,<:AbstractColumnMajor},
                                         <:AbstractStridedLayout, T}) where {UPLO,UNIT,T <: BlasFloat}
    A,x = M.A,M.B
    BLAS.trmv!(UPLO, 'N', UNIT, triangulardata(A), x)
end

@inline function materialize!(M::BlasMatLmulVec{<:TriangularLayout{UPLO,UNIT,<:AbstractRowMajor},
                                   <:AbstractStridedLayout, T}) where {UPLO,UNIT,T <: BlasFloat}
    A,x = M.A,M.B
    BLAS.trmv!(UPLO, 'T', UNIT, transpose(triangulardata(A)), x)
end


@inline function materialize!(M::BlasMatLmulVec{<:TriangularLayout{UPLO,UNIT,<:ConjLayout{<:AbstractRowMajor}},
                                   <:AbstractStridedLayout, T}) where {UPLO,UNIT,T <: BlasFloat}
    A,x = M.A,M.B
    BLAS.trmv!(UPLO, 'C', UNIT, triangulardata(A)', x)
end

# Triangular *\ Matrix

function materialize!(M::MatLmulMat{<:TriangularLayout})
    A,X = M.A,M.B
    size(A,2) == size(X,1) || thow(DimensionMismatch("Dimensions must match"))
    for j in axes(X,2)
        apply!(*, A, view(X,:,j))
    end
    X
end


