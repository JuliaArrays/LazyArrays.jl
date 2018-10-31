

struct Inv{Style, Typ}
    style::Style
    A::Typ
end

Inv(A) = Inv(MemoryLayout(A), A)
inv(A::Inv) = A.A

eltype(::Inv{<:Any,Typ}) where Typ = eltype(Typ)
eltype(::Type{<:Inv{<:Any,Typ}}) where Typ = eltype(Typ)

parent(A::Inv) = A.A

size(A::Inv) = reverse(size(parent(A)))
axes(A::Inv) = reverse(axes(parent(A)))
size(A::Inv,k) = size(A)[k]
axes(A::Inv,k) = axes(A)[k]


struct InverseLayout{ML} <: MemoryLayout
    layout::ML
end
MemoryLayout(Ai::Inv) = InverseLayout(MemoryLayout(Ai.A))


const Ldiv{StyleA, StyleB, AType, BType} =
    Mul2{<:InverseLayout{StyleA}, StyleB, <:Inv{StyleA,AType}, BType}
const ArrayLdivArray{styleA, styleB, p, q, T, V} =
    Ldiv{styleA, styleB, <:AbstractArray{T,p}, <:AbstractArray{V,q}}
const ArrayLdivArrayStyle{StyleA,StyleB,p,q} = ArrayMulArrayStyle{InverseLayout{StyleA}, StyleB, p, q}
const BArrayLdivArray{styleA, styleB, p, q, T, V} =
    Broadcasted{ArrayLdivArrayStyle{styleA,styleB,p,q}, <:Any, typeof(identity),
                <:Tuple{<:ArrayLdivArray{styleA,styleB,p,q,T,V}}}


BroadcastStyle(::Type{<:ArrayLdivArray{StyleA,StyleB,p,q}}) where {StyleA,StyleB,p,q} =
    ArrayLdivArrayStyle{StyleA,StyleB,p,q}()

Ldiv(A, B) = Mul(Inv(A), B)

macro lazyldiv(Typ)
    esc(quote
        LinearAlgebra.ldiv!(A::$Typ, x::AbstractVector) = (x .= LazyArrays.Ldiv(A,x))
        LinearAlgebra.ldiv!(A::$Typ, x::AbstractMatrix) = (x .= LazyArrays.Ldiv(A,x))
        LinearAlgebra.ldiv!(A::$Typ, x::StridedVector) = (x .= LazyArrays.Ldiv(A,x))
        LinearAlgebra.ldiv!(A::$Typ, x::StridedMatrix) = (x .= LazyArrays.Ldiv(A,x))

        Base.:\(A::$Typ, x::AbstractVector) = Inv(A) * x
        Base.:\(A::$Typ, x::AbstractMatrix) = Inv(A) * x
    end)
end

*(A::Inv, B) = materialize(Mul(A,B))

similar(A::Inv, ::Type{T}) where T = Array{T}(undef, size(A))
similar(M::ArrayLdivArray, ::Type{T}) where T = Array{T}(undef, size(M))

materialize(M::ArrayLdivArray) = copyto!(similar(M), M)

@inline function _copyto!(_, dest::AbstractArray, bc::BArrayLdivArray)
    (M,) = bc.args
    copyto!(dest, M)
end

function _copyto!(_, dest::AbstractArray, M::ArrayLdivArray)
    Ai, B = M.factors
    ldiv!(dest, factorize(inv(Ai)), B)
end

const MatLdivVec{styleA, styleB, T, V} = ArrayLdivArray{styleA, styleB, 2, 1, T, V}

broadcastable(M::MatLdivVec) = M


###
# Triangular
###

function _copyto!(_, dest::AbstractArray, M::ArrayLdivArray{<:TriangularLayout})
    Ai, B = M.factors
    dest .= B
    ldiv!(inv(Ai), dest)
end

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatLdivVec{<:TriangularLayout{UPLO,UNIT,<:AbstractColumnMajor},
                                   <:AbstractStridedLayout, T, T}) where {UPLO,UNIT,T <: BlasFloat}
    Ai,B = M.factors
    B ≡ dest || copyto!(dest, B)
    BLAS.trsv!(UPLO, 'N', UNIT, triangulardata(inv(Ai)), dest)
end

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatLdivVec{<:TriangularLayout{'U',UNIT,<:AbstractRowMajor},
                                   <:AbstractStridedLayout, T, T}) where {UNIT,T <: BlasFloat}
    Ai,B = M.factors
    B ≡ dest || copyto!(dest, B)
    BLAS.trsv!('L', 'T', UNIT, transpose(triangulardata(inv(Ai))), dest)
end

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatLdivVec{<:TriangularLayout{'L',UNIT,<:AbstractRowMajor},
                                   <:AbstractStridedLayout, T, T}) where {UNIT,T <: BlasFloat}
    Ai,B = M.factors
    B ≡ dest || copyto!(dest, B)
    BLAS.trsv!('U', 'T', UNIT, transpose(triangulardata(inv(Ai))), dest)
end


@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatLdivVec{T, <:TriangularLayout{'U',UNIT,<:ConjLayout{<:AbstractRowMajor}},
                                   <:AbstractStridedLayout, T, T}) where {UNIT,T <: BlasFloat}
    Ai,B = M.factors
    B ≡ dest || copyto!(dest, B)
    BLAS.trsv!('L', 'C', UNIT, triangulardata(inv(Ai))', dest)
end

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatLdivVec{<:TriangularLayout{'L',UNIT,<:ConjLayout{<:AbstractRowMajor}},
                                   <:AbstractStridedLayout, T, T}) where {UNIT,T <: BlasFloat}
    Ai,B = M.factors
    B ≡ dest || copyto!(dest, B)
    BLAS.trsv!('U', 'C', UNIT, triangulardata(inv(Ai))', dest)
end
