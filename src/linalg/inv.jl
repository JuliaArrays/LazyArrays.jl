

const PInv{Style, Typ} = Applied{LayoutApplyStyle{Tuple{Style}}, typeof(pinv), <:Tuple{Typ}}
const Inv{Style, Typ} = Applied{LayoutApplyStyle{Tuple{Style}}, typeof(inv), <:Tuple{Typ}}

Inv(A) = applied(inv, A)
PInv(A) = applied(pinv, A)

ApplyStyle(::typeof(inv), A::AbstractMatrix) = LayoutApplyStyle((MemoryLayout(A),))
ApplyStyle(::typeof(pinv), A::AbstractMatrix) = LayoutApplyStyle((MemoryLayout(A),))

const InvOrPInv = Union{Inv, PInv}

parent(A::PInv) = first(A.args)
parent(A::Inv) = first(A.args)

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



const Ldiv{StyleA, StyleB, AType, BType} =
    Applied{LayoutApplyStyle{Tuple{StyleA, StyleB}}, typeof(\), <:Tuple{AType, BType}}

Ldiv(A, B) = applied(\, A, B)

ApplyStyle(::typeof(\), A::AbstractArray, B::AbstractArray) =
    LayoutApplyStyle((MemoryLayout(A), MemoryLayout(B)))

size(L::Ldiv{<:Any,<:Any,<:Any,<:AbstractMatrix}) =
    (size(L.args[1], 2),size(L.args[2],2))
size(L::Ldiv{<:Any,<:Any,<:Any,<:AbstractVector}) =
    (size(L.args[1], 2),)
length(L::Ldiv{<:Any,<:Any,<:Any,<:AbstractVector}) =
    size(L.args[1], 2)

ndims(L::Applied{<:Any, typeof(\)}) = ndims(last(L.args))
eltype(M::Applied{<:Any, typeof(\)}) = promote_type(Base.promote_op(inv, eltype(first(M.args))),
                                                    eltype(last(M.args)))

struct ArrayLdivArrayStyle{StyleA, StyleB, p, q} <: BroadcastStyle end

@inline copyto!(dest::AbstractArray, bc::Broadcasted{<:ArrayLdivArrayStyle}) =
    _copyto!(MemoryLayout(dest), dest, bc)

const ArrayLdivArray{styleA, styleB, p, q, T, V} =
    Ldiv{styleA, styleB, <:AbstractArray{T,p}, <:AbstractArray{V,q}}
const BArrayLdivArray{styleA, styleB, p, q, T, V} =
    Broadcasted{ArrayLdivArrayStyle{styleA,styleB,p,q}, <:Any, typeof(identity),
                <:Tuple{<:ArrayLdivArray{styleA,styleB,p,q,T,V}}}


BroadcastStyle(::Type{<:ArrayLdivArray{StyleA,StyleB,p,q}}) where {StyleA,StyleB,p,q} =
    ArrayLdivArrayStyle{StyleA,StyleB,p,q}()


similar(A::InvOrPInv, ::Type{T}) where T = Array{T}(undef, size(A))
similar(A::Ldiv, ::Type{T}) where T = Array{T}(undef, size(A))
similar(M::ArrayLdivArray, ::Type{T}) where T = Array{T}(undef, size(M))

materialize(M::ArrayLdivArray) = copyto!(similar(M), M)

@inline function _copyto!(_, dest::AbstractArray, bc::BArrayLdivArray)
    (M,) = bc.args
    copyto!(dest, M)
end

if VERSION ≥ v"1.1-pre"
    function _copyto!(_, dest::AbstractArray, M::ArrayLdivArray)
        A, B = M.args
        ldiv!(dest, factorize(A), B)
    end
else
    function _copyto!(_, dest::AbstractArray, M::ArrayLdivArray)
        A, B = M.args
        ldiv!(dest, factorize(A), copy(B))
    end
end

const MatLdivVec{styleA, styleB, T, V} = ArrayLdivArray{styleA, styleB, 2, 1, T, V}
const MatLdivMat{styleA, styleB, T, V} = ArrayLdivArray{styleA, styleB, 2, 2, T, V}



###
# Triangular
###

function _copyto!(_, dest::AbstractArray, M::ArrayLdivArray{<:TriangularLayout})
    A, B = M.args
    dest ≡ B || (dest .= B)
    ldiv!(A, dest)
end

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatLdivVec{<:TriangularLayout{UPLO,UNIT,<:AbstractColumnMajor},
                                   <:AbstractStridedLayout, T, T}) where {UPLO,UNIT,T <: BlasFloat}
    A,B = M.args
    B ≡ dest || copyto!(dest, B)
    BLAS.trsv!(UPLO, 'N', UNIT, triangulardata(A), dest)
end

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatLdivVec{<:TriangularLayout{'U',UNIT,<:AbstractRowMajor},
                                   <:AbstractStridedLayout, T, T}) where {UNIT,T <: BlasFloat}
    A,B = M.args
    B ≡ dest || copyto!(dest, B)
    BLAS.trsv!('L', 'T', UNIT, transpose(triangulardata(A)), dest)
end

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatLdivVec{<:TriangularLayout{'L',UNIT,<:AbstractRowMajor},
                                   <:AbstractStridedLayout, T, T}) where {UNIT,T <: BlasFloat}
    A,B = M.args
    B ≡ dest || copyto!(dest, B)
    BLAS.trsv!('U', 'T', UNIT, transpose(triangulardata(A)), dest)
end


@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatLdivVec{T, <:TriangularLayout{'U',UNIT,<:ConjLayout{<:AbstractRowMajor}},
                                   <:AbstractStridedLayout, T, T}) where {UNIT,T <: BlasFloat}
    A,B = M.args
    B ≡ dest || copyto!(dest, B)
    BLAS.trsv!('L', 'C', UNIT, triangulardata(A)', dest)
end

@inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector{T},
         M::MatLdivVec{<:TriangularLayout{'L',UNIT,<:ConjLayout{<:AbstractRowMajor}},
                                   <:AbstractStridedLayout, T, T}) where {UNIT,T <: BlasFloat}
    A,B = M.args
    B ≡ dest || copyto!(dest, B)
    BLAS.trsv!('U', 'C', UNIT, triangulardata(A)', dest)
end

function _copyto!(_, dest::AbstractMatrix, M::MatLdivMat{<:TriangularLayout})
    A,X = M.args
    size(dest,2) == size(X,2) || thow(DimensionMismatch("Dimensions must match"))
    @views for j in axes(dest,2)
        dest[:,j] .= Ldiv(A, X[:,j])
    end
    dest
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


@inline function _copyto!(_, dest::AbstractArray, M::MatMulVec{<:ApplyLayout{typeof(inv)}})
    Ai,b = M.args
    dest .= Ldiv(parent(Ai.applied), b)
end

@inline function _copyto!(_, dest::AbstractArray, M::MatMulVec{<:ApplyLayout{typeof(pinv)}})
    Ai,b = M.args
    dest .= Ldiv(parent(Ai.applied), b)
end
