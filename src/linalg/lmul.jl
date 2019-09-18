struct Lmul{StyleA, StyleB, TypeA, TypeB}
    A::TypeA
    B::TypeB
end

Lmul(A::TypeA, B::TypeB) where {TypeA,TypeB} = Lmul{typeof(MemoryLayout(TypeA)),typeof(MemoryLayout(TypeB)),TypeA,TypeB}(A,B)

BroadcastStyle(::Type{<:Lmul}) = ApplyBroadcastStyle()


struct LmulStyle <: AbstractArrayApplyStyle end

broadcastable(M::Lmul) = M


const MatLmulVec{StyleA,StyleB} = Lmul{StyleA,StyleB,<:AbstractMatrix,<:AbstractVector}
const MatLmulMat{StyleA,StyleB} = Lmul{StyleA,StyleB,<:AbstractMatrix,<:AbstractMatrix}

const BlasMatLmulVec{StyleA,StyleB,T<:BlasFloat} = Lmul{StyleA,StyleB,<:AbstractMatrix{T},<:AbstractVector{T}}
const BlasMatLmulMat{StyleA,StyleB,T<:BlasFloat} = Lmul{StyleA,StyleB,<:AbstractMatrix{T},<:AbstractMatrix{T}}

####
# LMul materialize
####

Lmul(M::Mul) = Lmul(M.args...)

eltype(::Lmul{<:Any,<:Any,A,B}) where {A,B} = promote_type(eltype(A), eltype(B))
size(M::Lmul, p::Int) = size(M)[p]
axes(M::Lmul, p::Int) = axes(M)[p]
length(M::Lmul) = prod(size(M))
size(M::Lmul) = map(length,axes(M))
axes(M::MatLmulVec) = (axes(M.A,1),)
axes(M::Lmul) = (axes(M.A,1),axes(M.B,2))


similar(M::Lmul, ::Type{T}, axes) where {T,N} = similar(Array{T}, axes)
similar(M::Lmul, ::Type{T}) where T = similar(M, T, axes(M))
similar(M::Lmul) = similar(M, eltype(M))

similar(M::Applied{LmulStyle}, ::Type{T}) where T = similar(Lmul(M), T)
copy(M::Applied{LmulStyle}) = copy(Lmul(M))


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


####
# Diagonal
####

# combine_mul_styles(::DiagonalLayout) = LmulStyle()

# Diagonal multiplication never changes structure
similar(M::Lmul{<:DiagonalLayout}, ::Type{T}, axes) where T = similar(M.B, T, axes)
# equivalent to rescaling
function materialize!(M::Lmul{<:DiagonalLayout{<:AbstractFillLayout}})
    M.B .= getindex_value(M.A.diag) .* M.B
    M.B
end

copy(M::Lmul{<:DiagonalLayout}) = M.A.diag .* M.B
