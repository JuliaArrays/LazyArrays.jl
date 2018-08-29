

struct Inv{T, Style, Typ}
    style::Style
    A::Typ
end

Inv(style::S, A::T) where {S,T} = Inv{eltype(A),S,T}(style, A)
Inv(A) = Inv(MemoryLayout(A), A)
inv(A::Inv) = A.A

eltype(::Inv{T}) where T = T
eltype(::Type{<:Inv{T}}) where T = T

struct InverseLayout{ML} <: MemoryLayout
    layout::ML
end
MemoryLayout(Ai::Inv) = InverseLayout(MemoryLayout(Ai.A))


const Ldiv{T, StyleA, StyleB, AType, BType} =
    Mul{T, InverseLayout{StyleA}, StyleB, <:Inv{T,StyleA,AType}, BType}
const ArrayLdivArray{TV, styleA, styleB, p, q, T, V} =
    Ldiv{TV, styleA, styleB, <:AbstractArray{T,p}, <:AbstractArray{V,q}}
const ArrayLdivArrayStyle{StyleA,StyleB,p,q} = ArrayMulArrayStyle{InverseLayout{StyleA}, StyleB, p, q}
const BArrayLdivArray{TV, styleA, styleB, p, q, T, V} =
    Broadcasted{ArrayLdivArrayStyle{styleA,styleB,p,q}, <:Any, typeof(identity),
                <:Tuple{<:ArrayLdivArray{TV,styleA,styleB,p,q,T,V}}}


BroadcastStyle(::Type{<:ArrayLdivArray{<:Any,StyleA,StyleB,p,q}}) where {StyleA,StyleB,p,q} =
    ArrayLdivArrayStyle{StyleA,StyleB,p,q}()

Ldiv(A, B) = Mul(Inv(A), B)

macro lazyldiv(Typ)
    esc(quote
        LinearAlgebra.ldiv!(A::$Typ, x::AbstractVector) = (x .= Ldiv(A,x))
        LinearAlgebra.ldiv!(A::$Typ, x::AbstractMatrix) = (x .= Ldiv(A,x))
        LinearAlgebra.ldiv!(A::$Typ, x::StridedVector) = (x .= Ldiv(A,x))
        LinearAlgebra.ldiv!(A::$Typ, x::StridedMatrix) = (x .= Ldiv(A,x))
    end)
end

@inline function _copyto!(_, dest::AbstractArray, bc::BArrayLdivArray)
    (M,) = bc.args
    copyto!(dest, M)
end

function _copyto!(_, dest::AbstractArray, M::ArrayLdivArray)
    A,x = inv(M.A), M.B
    ldiv!(dest, factorize(A), x)
end

let (p,q) = (2,1)
    global const MatLdivVec{TV, styleA, styleB, T, V} = ArrayLdivArray{TV, styleA, styleB, p, q, T, V}
end

broadcastable(M::MatLdivVec) = M


###
# Triangular
###

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatLdivVec{T, <:TriangularLayout{UPLO,UNIT,<:AbstractColumnMajor},
                                   <:AbstractStridedLayout, T, T}) where {UPLO,UNIT,T <: BlasFloat}
    A,x = inv(M.A), M.B
    x ≡ dest || copyto!(dest, x)
    BLAS.trsv!(UPLO, 'N', UNIT, triangulardata(A), dest)
end

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatLdivVec{T, <:TriangularLayout{'U',UNIT,<:AbstractRowMajor},
                                   <:AbstractStridedLayout, T, T}) where {UNIT,T <: BlasFloat}
    A,x = inv(M.A), M.B
    x ≡ dest || copyto!(dest, x)
    BLAS.trsv!('L', 'T', UNIT, transpose(triangulardata(A)), dest)
end

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatLdivVec{T, <:TriangularLayout{'L',UNIT,<:AbstractRowMajor},
                                   <:AbstractStridedLayout, T, T}) where {UNIT,T <: BlasFloat}
    A,x = inv(M.A), M.B
    x ≡ dest || copyto!(dest, x)
    BLAS.trsv!('U', 'T', UNIT, transpose(triangulardata(A)), dest)
end


@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatLdivVec{T, <:TriangularLayout{'U',UNIT,<:ConjLayout{<:AbstractRowMajor}},
                                   <:AbstractStridedLayout, T, T}) where {UNIT,T <: BlasFloat}
    A,x = inv(M.A), M.B
    x ≡ dest || copyto!(dest, x)
    BLAS.trsv!('L', 'C', UNIT, triangulardata(A)', dest)
end

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatLdivVec{T, <:TriangularLayout{'L',UNIT,<:ConjLayout{<:AbstractRowMajor}},
                                   <:AbstractStridedLayout, T, T}) where {UNIT,T <: BlasFloat}
    A,x = inv(M.A), M.B
    x ≡ dest || copyto!(dest, x)
    BLAS.trsv!('U', 'C', UNIT, triangulardata(A)', dest)
end
