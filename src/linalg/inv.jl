abstract type AbstractPInv{Style, Typ} end

eltype(::AbstractPInv{<:Any,Typ}) where Typ = eltype(Typ)
eltype(::Type{<:AbstractPInv{<:Any,Typ}}) where Typ = eltype(Typ)

struct PInv{Style, Typ} <: AbstractPInv{Style, Typ}
    style::Style
    A::Typ
end
struct Inv{Style, Typ} <: AbstractPInv{Style, Typ}
    style::Style
    A::Typ
    function Inv{Style,Typ}(style::Style, A::Typ) where {Style,Typ}
        checksquare(A)
        new{Style,Typ}(style,A)
    end
end

Inv(style::Style, A::Typ) where {Style,Typ} = Inv{Style,Typ}(style, A)

PInv(A) = PInv(MemoryLayout(A), A)
Inv(A) = Inv(MemoryLayout(A), A)



pinv(A::PInv) = A.A
function inv(A::PInv)
    checksquare(A.A)
    A.A
end

inv(A::Inv) = A.A
pinv(A::Inv) = inv(A)


parent(A::AbstractPInv) = A.A


size(A::AbstractPInv) = reverse(size(parent(A)))
axes(A::AbstractPInv) = reverse(axes(parent(A)))
size(A::AbstractPInv, k) = size(A)[k]
axes(A::AbstractPInv, k) = axes(A)[k]


abstract type AbstractPInvLayout{ML} <: MemoryLayout end

# struct InverseLayout{ML} <: AbstractPInvLayout{ML}
#     layout::ML
# end

struct PInvLayout{ML} <: AbstractPInvLayout{ML}
    layout::ML
end

MemoryLayout(Ai::AbstractPInv) = PInvLayout(MemoryLayout(Ai.A))


const Ldiv{StyleA, StyleB, AType, BType} =
    Mul2{<:AbstractPInvLayout{StyleA}, StyleB, <:AbstractPInv{StyleA,AType}, BType}
const ArrayLdivArray{styleA, styleB, p, q, T, V} =
    Ldiv{styleA, styleB, <:AbstractArray{T,p}, <:AbstractArray{V,q}}
const ArrayLdivArrayStyle{StyleA,StyleB,p,q} =
    ArrayMulArrayStyle{PInvLayout{StyleA}, StyleB, p, q}
const BArrayLdivArray{styleA, styleB, p, q, T, V} =
    Broadcasted{ArrayLdivArrayStyle{styleA,styleB,p,q}, <:Any, typeof(identity),
                <:Tuple{<:ArrayLdivArray{styleA,styleB,p,q,T,V}}}


BroadcastStyle(::Type{<:ArrayLdivArray{StyleA,StyleB,p,q}}) where {StyleA,StyleB,p,q} =
    ArrayLdivArrayStyle{StyleA,StyleB,p,q}()
broadcastable(M::ArrayLdivArray) = M

Ldiv(A, B) = Mul(PInv(A), B)

macro lazyldiv(Typ)
    esc(quote
        LinearAlgebra.ldiv!(A::$Typ, x::AbstractVector) = (x .= LazyArrays.Ldiv(A,x))
        LinearAlgebra.ldiv!(A::$Typ, x::AbstractMatrix) = (x .= LazyArrays.Ldiv(A,x))
        LinearAlgebra.ldiv!(A::$Typ, x::StridedVector) = (x .= LazyArrays.Ldiv(A,x))
        LinearAlgebra.ldiv!(A::$Typ, x::StridedMatrix) = (x .= LazyArrays.Ldiv(A,x))

        Base.:\(A::$Typ, x::AbstractVector) = PInv(A) * x
        Base.:\(A::$Typ, x::AbstractMatrix) = PInv(A) * x
    end)
end

*(A::AbstractPInv, B, C...) = materialize(Mul(A,B, C...))
*(A::AbstractPInv, B::Mul) = materialize(Mul(A, B.args...))

similar(A::AbstractPInv, ::Type{T}) where T = Array{T}(undef, size(A))
similar(M::ArrayLdivArray, ::Type{T}) where T = Array{T}(undef, size(M))

materialize(M::ArrayLdivArray) = copyto!(similar(M), M)

@inline function _copyto!(_, dest::AbstractArray, bc::BArrayLdivArray)
    (M,) = bc.args
    copyto!(dest, M)
end

if VERSION ≥ v"1.1-pre"
    function _copyto!(_, dest::AbstractArray, M::ArrayLdivArray)
        Ai, B = M.args
        ldiv!(dest, factorize(pinv(Ai)), B)
    end
else
    function _copyto!(_, dest::AbstractArray, M::ArrayLdivArray)
        Ai, B = M.args
        ldiv!(dest, factorize(pinv(Ai)), copy(B))
    end
end

const MatLdivVec{styleA, styleB, T, V} = ArrayLdivArray{styleA, styleB, 2, 1, T, V}
const MatLdivMat{styleA, styleB, T, V} = ArrayLdivArray{styleA, styleB, 2, 2, T, V}

broadcastable(M::MatLdivVec) = M


###
# Triangular
###

function _copyto!(_, dest::AbstractArray, M::ArrayLdivArray{<:TriangularLayout})
    Ai, B = M.args
    dest ≡ B || (dest .= B)
    ldiv!(pinv(Ai), dest)
end

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatLdivVec{<:TriangularLayout{UPLO,UNIT,<:AbstractColumnMajor},
                                   <:AbstractStridedLayout, T, T}) where {UPLO,UNIT,T <: BlasFloat}
    Ai,B = M.args
    B ≡ dest || copyto!(dest, B)
    BLAS.trsv!(UPLO, 'N', UNIT, triangulardata(pinv(Ai)), dest)
end

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatLdivVec{<:TriangularLayout{'U',UNIT,<:AbstractRowMajor},
                                   <:AbstractStridedLayout, T, T}) where {UNIT,T <: BlasFloat}
    Ai,B = M.args
    B ≡ dest || copyto!(dest, B)
    BLAS.trsv!('L', 'T', UNIT, transpose(triangulardata(pinv(Ai))), dest)
end

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatLdivVec{<:TriangularLayout{'L',UNIT,<:AbstractRowMajor},
                                   <:AbstractStridedLayout, T, T}) where {UNIT,T <: BlasFloat}
    Ai,B = M.args
    B ≡ dest || copyto!(dest, B)
    BLAS.trsv!('U', 'T', UNIT, transpose(triangulardata(pinv(Ai))), dest)
end


@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatLdivVec{T, <:TriangularLayout{'U',UNIT,<:ConjLayout{<:AbstractRowMajor}},
                                   <:AbstractStridedLayout, T, T}) where {UNIT,T <: BlasFloat}
    Ai,B = M.args
    B ≡ dest || copyto!(dest, B)
    BLAS.trsv!('L', 'C', UNIT, triangulardata(pinv(Ai))', dest)
end

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatLdivVec{<:TriangularLayout{'L',UNIT,<:ConjLayout{<:AbstractRowMajor}},
                                   <:AbstractStridedLayout, T, T}) where {UNIT,T <: BlasFloat}
    Ai,B = M.args
    B ≡ dest || copyto!(dest, B)
    BLAS.trsv!('U', 'C', UNIT, triangulardata(pinv(Ai))', dest)
end

function _copyto!(_, dest::AbstractMatrix, M::MatLdivMat{<:TriangularLayout})
    A,X = M.args
    size(dest,2) == size(X,2) || thow(DimensionMismatch("Dimensions must match"))
    @views for j in axes(dest,2)
        dest[:,j] .= Mul(A, X[:,j])
    end
    dest
end


function _PInvMatrix end

struct PInvMatrix{T, PINV<:AbstractPInv} <: AbstractMatrix{T}
    pinv::PINV
    global _PInvMatrix(pinv::PINV) where PINV = new{eltype(pinv), PINV}(pinv)
end

const InvMatrix{T, INV<:Inv} = PInvMatrix{T,INV}

PInvMatrix(M) = _PInvMatrix(PInv(M))
InvMatrix(M) = _PInvMatrix(Inv(M))

axes(A::PInvMatrix) = axes(A.pinv)
size(A::PInvMatrix) = map(length, axes(A))

@propagate_inbounds getindex(A::PInvMatrix{T}, k::Int, j::Int) where T =
    (A.pinv*[Zeros(j-1); one(T); Zeros(size(A,2) - j)])[k]

MemoryLayout(M::PInvMatrix) = MemoryLayout(M.pinv)
