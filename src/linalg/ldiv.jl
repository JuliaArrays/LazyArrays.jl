

const PInv{Style, Typ} = Applied{Style, typeof(pinv), <:Tuple{Typ}}
const Inv{Style, Typ} = Applied{Style, typeof(inv), <:Tuple{Typ}}

Inv(A) = applied(inv, A)
PInv(A) = applied(pinv, A)

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
eltype(A::InvOrPInv) = Base.promote_op(inv, eltype(parent(A)))

struct Ldiv{StyleA, StyleB, AType, BType}
    A::AType
    B::BType
end

Ldiv(A::AType, B::BType) where {AType,BType} = 
    Ldiv{typeof(MemoryLayout(AType)),typeof(MemoryLayout(BType)),AType,BType}(A, B)

struct LdivBroadcastStyle <: BroadcastStyle end
struct LdivApplyStyle <: ApplyStyle end


ldivapplystyle(_, _) = LdivApplyStyle()
ldivapplystyle(::LazyLayout, ::LazyLayout) = LazyArrayApplyStyle()
ldivapplystyle(::LazyLayout, _) = LazyArrayApplyStyle()
ldivapplystyle(_, ::LazyLayout) = LazyArrayApplyStyle()
ApplyStyle(::typeof(\), ::Type{A}, ::Type{B}) where {A<:AbstractArray,B<:AbstractArray} = 
    ldivapplystyle(MemoryLayout(A), MemoryLayout(B))

size(L::Ldiv{<:Any,<:Any,<:Any,<:AbstractMatrix}) = (size(L.A, 2),size(L.B,2))
size(L::Ldiv{<:Any,<:Any,<:Any,<:AbstractVector}) = (size(L.A, 2),)
axes(L::Ldiv{<:Any,<:Any,<:Any,<:AbstractMatrix}) = (axes(L.A, 2),axes(L.B,2))
axes(L::Ldiv{<:Any,<:Any,<:Any,<:AbstractVector}) = (axes(L.A, 2),)    
length(L::Ldiv{<:Any,<:Any,<:Any,<:AbstractVector}) =size(L.A, 2)

_ldivaxes(::Tuple{}, ::Tuple{}) = ()
_ldivaxes(::Tuple{}, Bax::Tuple) = Bax
_ldivaxes(::Tuple{<:Any}, ::Tuple{<:Any}) = ()
_ldivaxes(::Tuple{<:Any}, Bax::Tuple{<:Any,<:Any}) = (OneTo(1),last(Bax))
_ldivaxes(Aax::Tuple{<:Any,<:Any}, ::Tuple{<:Any}) = (last(Aax),)
_ldivaxes(Aax::Tuple{<:Any,<:Any}, Bax::Tuple{<:Any,<:Any}) = (last(Aax),last(Bax))

@inline ldivaxes(A, B) = _ldivaxes(axes(A), axes(B))

axes(M::Applied{Style,typeof(\)}) where Style = ldivaxes(M.args...)
axes(M::Applied{Style,typeof(\)}, p::Int)  where Style = axes(M)[p]
size(M::Applied{Style,typeof(\)}) where Style = length.(axes(M))

ndims(L::Ldiv) = ndims(last(L.args))
eltype(M::Ldiv) = promote_type(Base.promote_op(inv, eltype(M.A)), eltype(M.B))

@inline eltype(M::Applied{Style,typeof(\)}) where Style = eltype(Ldiv(M.args...))
@inline ndims(M::Applied{Style,typeof(\)}) where Style = ndims(last(M.args))

BroadcastStyle(::Type{<:Ldiv}) = ApplyBroadcastStyle()
broadcastable(M::Ldiv) = M

similar(A::Ldiv, ::Type{T}) where T = similar(Array{T}, axes(A))
similar(A::Ldiv) = similar(A, eltype(A))

function instantiate(L::Ldiv)
    check_ldiv_axes(L.A, L.B)
    Ldiv(instantiate(L.A), instantiate(L.B))
end


check_ldiv_axes(A, B) =
    axes(A,1) == axes(B,1) || throw(DimensionMismatch("First axis of A, $(axes(A,1)), and first axis of B, $(axes(B,1)) must match"))

check_applied_axes(A::Applied{<:Any,typeof(\)}) = check_ldiv_axes(A.args...)

copy(M::Ldiv) = copyto!(similar(M), M)
materialize(M::Ldiv) = copy(instantiate(M))



_ldiv!(A, B) = ldiv!(factorize(A), B)
_ldiv!(A::Factorization, B) = ldiv!(A, B)

_ldiv!(dest, A, B) = ldiv!(dest, factorize(A), B)
_ldiv!(dest, A::Factorization, B) = ldiv!(dest, A, B)


materialize!(M::Ldiv) = _ldiv!(M.A, M.B)
if VERSION ≥ v"1.1-pre"
    copyto!(dest::AbstractArray, M::Ldiv) = _ldiv!(dest, M.A, M.B)
else
    copyto!(dest::AbstractArray, M::Ldiv) = _ldiv!(dest, M.A, copy(M.B))
end

const MatLdivVec{styleA, styleB, T, V} = Ldiv{styleA, styleB, <:AbstractMatrix{T}, <:AbstractVector{V}}
const MatLdivMat{styleA, styleB, T, V} = Ldiv{styleA, styleB, <:AbstractMatrix{T}, <:AbstractMatrix{V}}
const BlasMatLdivVec{styleA, styleB, T<:BlasFloat} = MatLdivVec{styleA, styleB, T, T}
const BlasMatLdivMat{styleA, styleB, T<:BlasFloat} = MatLdivMat{styleA, styleB, T, T}


######
# PInv/Inv
########


const PInvMatrix{T,Arg} = ApplyMatrix{T,typeof(pinv),<:Tuple{Arg}}
const InvMatrix{T,Arg} = ApplyMatrix{T,typeof(inv),<:Tuple{Arg}}

PInvMatrix(A) = ApplyMatrix(pinv, A)
function InvMatrix(A)
    checksquare(A)
    ApplyMatrix(inv, A)
end

parent(A::PInvMatrix) = first(A.args)
parent(A::InvMatrix) = first(A.args)
axes(A::PInvMatrix) = reverse(axes(parent(A)))
size(A::PInvMatrix) = map(length, axes(A))

@propagate_inbounds getindex(A::PInvMatrix{T}, k::Int, j::Int) where T =
    (parent(A)\[Zeros(j-1); one(T); Zeros(size(A,2) - j)])[k]

@propagate_inbounds getindex(A::InvMatrix{T}, k::Int, j::Int) where T =
    (parent(A)\[Zeros(j-1); one(T); Zeros(size(A,2) - j)])[k]

struct InvLayout{L} <: MemoryLayout end
struct PInvLayout{L} <: MemoryLayout end

applylayout(::Type{typeof(inv)}, ::A) where A = InvLayout{A}()
applylayout(::Type{typeof(pinv)}, ::A) where A = PInvLayout{A}()

mulapplystyle(::InvLayout{A}, B) where A = ldivapplystyle(A, B)
mulapplystyle(::PInvLayout{A}, B) where A = ldivapplystyle(A, B)


@inline function Ldiv(M::Mul)
    Ai,b = M.args
    Ldiv(parent(Ai), b)
end
Ldiv(A::Applied{<:Any,typeof(\)}) = Ldiv(A.args...)


similar(M::Applied{LdivApplyStyle}, ::Type{T}) where T = similar(Ldiv(M), T)
copy(M::Applied{LdivApplyStyle}) = copy(Ldiv(M))
@inline copyto!(dest::AbstractArray, M::Applied{LdivApplyStyle}) = copyto!(dest, Ldiv(M))
@inline materialize!(M::Applied{LdivApplyStyle}) = materialize!(Ldiv(M))

@propagate_inbounds getindex(A::Applied{LazyArrayApplyStyle,typeof(\)}, kj...) = 
    materialize(Ldiv(A))[kj...]



