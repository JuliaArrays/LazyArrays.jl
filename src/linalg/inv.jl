

const PInv{Style, Typ} = Applied{Style, typeof(pinv), <:Tuple{Typ}}
const Inv{Style, Typ} = Applied{Style, typeof(inv), <:Tuple{Typ}}

Inv(A) = applied(inv, A)
PInv(A) = applied(pinv, A)

ApplyStyle(::typeof(inv), A::AbstractMatrix) = LayoutApplyStyle((MemoryLayout(A),))
ApplyStyle(::typeof(pinv), A::AbstractMatrix) = LayoutApplyStyle((MemoryLayout(A),))

const InvOrPInv = Union{PInv, Inv}

parent(A::InvOrPInv) = first(A.args)

pinv(A::PInv) = parent(A)
function inv(A::PInv)
    checksquare(parent(A))
    parent(A)
end

inv(A::Inv) = parent(A)
pinv(A::Inv) = inv(A)

ndims(A::InvOrPInv) = ndims(parent(A))




size(A::InvOrPInv) = reverse(size(parent(A)))
axes(A::InvOrPInv) = reverse(axes(parent(A)))
size(A::InvOrPInv, k) = size(A)[k]
axes(A::InvOrPInv, k) = axes(A)[k]
eltype(A::InvOrPInv) = eltype(parent(A))


struct Ldiv{StyleA, StyleB, AType, BType}
    A::AType
    B::BType
end

Ldiv(A::AType, B::BType) where {AType,BType} = 
    Ldiv{typeof(MemoryLayout(AType)),typeof(MemoryLayout(BType)),AType,BType}(A, B)

struct LdivBroadcastStyle <: BroadcastStyle end
struct LdivApplyStyle <: ApplyStyle end

ApplyStyle(::typeof(\), ::Type{<:AbstractArray}, ::Type{<:AbstractArray}) = LdivApplyStyle()

size(L::Ldiv{<:Any,<:Any,<:Any,<:AbstractMatrix}) = (size(L.A, 2),size(L.B,2))
size(L::Ldiv{<:Any,<:Any,<:Any,<:AbstractVector}) = (size(L.A, 2),)
axes(L::Ldiv{<:Any,<:Any,<:Any,<:AbstractMatrix}) = (axes(L.A, 2),axes(L.B,2))
axes(L::Ldiv{<:Any,<:Any,<:Any,<:AbstractVector}) = (axes(L.A, 2),)    
length(L::Ldiv{<:Any,<:Any,<:Any,<:AbstractVector}) =size(L.A, 2)

ndims(L::Ldiv) = ndims(last(L.args))
eltype(M::Ldiv) = promote_type(Base.promote_op(inv, eltype(M.A)), eltype(M.B))

BroadcastStyle(::Type{<:Ldiv}) = ApplyBroadcastStyle()
broadcastable(M::Ldiv) = M


similar(A::InvOrPInv, ::Type{T}) where T = Array{T}(undef, size(A))
similar(A::Ldiv, ::Type{T}) where T = Array{T}(undef, size(A))
similar(A::Ldiv) = similar(A, eltype(A))

materialize(M::Ldiv) = copyto!(similar(M), M)

if VERSION ≥ v"1.1-pre"
    copyto!(dest::AbstractArray, M::Ldiv) = ldiv!(dest, factorize(M.A), M.B)
else
    copyto!(dest::AbstractArray, M::Ldiv) = ldiv!(dest, factorize(M.A), copy(M.B))
end

const MatLdivVec{styleA, styleB, T, V} = Ldiv{styleA, styleB, <:AbstractMatrix{T}, <:AbstractVector{V}}
const MatLdivMat{styleA, styleB, T, V} = Ldiv{styleA, styleB, <:AbstractMatrix{T}, <:AbstractMatrix{V}}
const BlasMatLdivVec{styleA, styleB, T<:BlasFloat} = MatLdivVec{styleA, styleB, T, T}
const BlasMatLdivMat{styleA, styleB, T<:BlasFloat} = MatLdivMat{styleA, styleB, T, T}



###
# Triangular
###

@inline function copyto!(dest::AbstractArray, M::Ldiv{<:TriangularLayout})
    A, B = M.A, M.B
    dest ≡ B || (dest .= B)
    materialize!(Ldiv(A, dest))
end

materialize!(M::Ldiv) = ldiv!(M.A, M.B)

@inline materialize!(M::BlasMatLdivVec{<:TriangularLayout{UPLO,UNIT,<:AbstractColumnMajor},
                                       <:AbstractStridedLayout}) where {UPLO,UNIT} =
    BLAS.trsv!(UPLO, 'N', UNIT, triangulardata(M.A), M.B)

@inline materialize!(M::BlasMatLdivVec{<:TriangularLayout{'U',UNIT,<:AbstractRowMajor},
                                                <:AbstractStridedLayout}) where {UNIT} =
    BLAS.trsv!('L', 'T', UNIT, transpose(triangulardata(M.A)), M.B)

@inline materialize!(M::BlasMatLdivVec{<:TriangularLayout{'L',UNIT,<:AbstractRowMajor},
                                                <:AbstractStridedLayout}) where {UNIT} =
    BLAS.trsv!('U', 'T', UNIT, transpose(triangulardata(M.A)), M.B)


@inline materialize!(M::BlasMatLdivVec{<:TriangularLayout{'U',UNIT,<:ConjLayout{<:AbstractRowMajor}},
                                                <:AbstractStridedLayout}) where {UNIT} =
    BLAS.trsv!('L', 'C', UNIT, triangulardata(M.A)', M.B)

@inline materialize!(M::BlasMatLdivVec{<:TriangularLayout{'L',UNIT,<:ConjLayout{<:AbstractRowMajor}},
                                                <:AbstractStridedLayout}) where {UNIT,T} =
    BLAS.trsv!('U', 'C', UNIT, triangulardata(M.A)', M.B)

function materialize!(M::MatLdivMat{<:TriangularLayout})
    A,X = M.A,M.B
    size(A,2) == size(X,1) || thow(DimensionMismatch("Dimensions must match"))
    @views for j in axes(X,2)
        materialize!(Ldiv(A, X[:,j]))
    end
    X
end


const PInvMatrix{T,App<:PInv} = ApplyMatrix{T,App}
const InvMatrix{T,App<:Inv} = ApplyMatrix{T,App}

PInvMatrix(A) = ApplyMatrix(pinv, A)
function InvMatrix(A)
    checksquare(A)
    ApplyMatrix(inv, A)
end

axes(A::PInvMatrix) = reverse(axes(parent(A.applied)))
size(A::PInvMatrix) = map(length, axes(A))

@propagate_inbounds getindex(A::PInvMatrix{T}, k::Int, j::Int) where T =
    (parent(A.applied)\[Zeros(j-1); one(T); Zeros(size(A,2) - j)])[k]

@propagate_inbounds getindex(A::InvMatrix{T}, k::Int, j::Int) where T =
    (parent(A.applied)\[Zeros(j-1); one(T); Zeros(size(A,2) - j)])[k]

mulapplystyle(::ApplyLayout{typeof(inv)}, _) = LdivApplyStyle()
mulapplystyle(::ApplyLayout{typeof(pinv)}, _) = LdivApplyStyle()

similar(M::Applied{LdivApplyStyle}, ::Type{T}, ::NTuple{N,OneTo{Int}}) where {T,N} = Array{T}(undef, size(M))
similar(M::Applied{LdivApplyStyle}, ::Type{T}) where T = similar(M, T, axes(M))

materialize(A::Mul{LdivApplyStyle}) = _materialize(A, axes(A))
_materialize(A::Applied{LdivApplyStyle}, _) = copyto!(similar(A), A)


@inline function materialize!(M::Mul{LdivApplyStyle})
    Ai,b = M.args
    materialize!(Ldiv(parent(Ai.applied), b))
end

@inline function copyto!(dest::AbstractArray, M::Mul{LdivApplyStyle})
    Ai,b = M.args
    dest .= Ldiv(parent(Ai.applied), b)
end

@inline function materialize!(M::Applied{LdivApplyStyle,typeof(\)})
    Ai,b = M.args
    materialize!(Ldiv(parent(Ai.applied), b))
end

@inline function copyto!(dest::AbstractArray, M::Applied{LdivApplyStyle,typeof(\)})
    Ai,b = M.args
    dest .= Ldiv(parent(Ai.applied), b)
end