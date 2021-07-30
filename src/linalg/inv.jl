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

# Use ArrayLayouts.ldiv instead of \
struct LdivStyle <: ApplyStyle end
struct RdivStyle <: ApplyStyle end

ApplyStyle(::typeof(\), ::Type{A}, ::Type{B}) where {A<:AbstractArray,B<:AbstractArray} = LdivStyle()
ApplyStyle(::typeof(/), ::Type{A}, ::Type{B}) where {A<:AbstractArray,B<:AbstractArray} = RdivStyle()


axes(M::Applied{Style,typeof(\)}) where Style = ldivaxes(M.args...)
axes(M::Applied{Style,typeof(\)}, p::Int)  where Style = axes(M)[p]
size(M::Applied{Style,typeof(\)}) where Style = length.(axes(M))
@inline eltype(M::Applied{Style,typeof(\)}) where Style = eltype(Ldiv(M.args...))
@inline ndims(M::Applied{Style,typeof(\)}) where Style = ndims(last(M.args))


axes(M::Applied{Style,typeof(/)}) where Style = axes(Rdiv(M.args...))
axes(M::Applied{Style,typeof(/)}, p::Int)  where Style = axes(M)[p]
size(M::Applied{Style,typeof(/)}) where Style = length.(axes(M))
@inline eltype(M::Applied{Style,typeof(/)}) where Style = eltype(Rdiv(M.args...))
@inline ndims(M::Applied{Style,typeof(/)}) where Style = ndims(first(M.args))


check_applied_axes(A::Applied{<:Any,typeof(\)}) = check_ldiv_axes(A.args...)

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
inv(A::InvMatrix) = parent(A)
pinv(A::InvMatrix) = parent(A)
pinv(A::PInvMatrix) = parent(A)



abstract type AbstractInvLayout{L} <: MemoryLayout end
struct InvLayout{L} <: AbstractInvLayout{L} end
struct PInvLayout{L} <: AbstractInvLayout{L} end

applylayout(::Type{typeof(inv)}, ::A) where A = InvLayout{A}()
applylayout(::Type{typeof(pinv)}, ::A) where A = PInvLayout{A}()

# Can always  simplify by lowering to \
simplifiable(::Mul{<:AbstractInvLayout}) = Val(true)

copy(M::Mul{<:AbstractInvLayout}) = ArrayLayouts.ldiv(pinv(M.A), M.B)
copy(M::Mul{<:AbstractInvLayout,<:AbstractLazyLayout}) = ArrayLayouts.ldiv(pinv(M.A), M.B)
@inline copy(M::Mul{<:AbstractInvLayout,ApplyLayout{typeof(*)}}) = simplify(M)
copy(L::Ldiv{<:AbstractInvLayout}) = pinv(L.A) * L.B
Ldiv(A::Applied{<:Any,typeof(\)}) = Ldiv(A.args...)


similar(M::Applied{LdivStyle}, ::Type{T}) where T = similar(Ldiv(M), T)
@inline copy(M::Applied{LdivStyle}) = ldiv(arguments(M)...)
@inline copyto!(dest::AbstractArray, M::Applied{LdivStyle}) = copyto!(dest, Ldiv(M))
@inline materialize!(M::Applied{LdivStyle}) = materialize!(Ldiv(M))

@propagate_inbounds getindex(A::Applied{<:Any,typeof(\)}, kj...) = Ldiv(A)[kj...]


###
# * layout
###
@inline function _copy_ldiv_mul(A, B₀, B₁...)
    AB₀ = A \  B₀
    MemoryLayout(AB₀) isa ApplyLayout{typeof(\)} && return lazymaterialize(*, AB₀, B₁...)
    apply(*, AB₀, B₁...)
end
@inline copy(L::Ldiv{<:DiagonalLayout,ApplyLayout{typeof(*)}}) = _copy_ldiv_mul(L.A, arguments(ApplyLayout{typeof(*)}(), L.B)...)
@inline copy(L::Ldiv{<:Any,ApplyLayout{typeof(*)}}) = _copy_ldiv_mul(L.A, arguments(ApplyLayout{typeof(*)}(), L.B)...)
@inline copy(L::Ldiv{<:AbstractLazyLayout,ApplyLayout{typeof(*)}}) = _copy_ldiv_mul(L.A, arguments(ApplyLayout{typeof(*)}(), L.B)...)
@inline copy(L::Ldiv{ApplyLayout{typeof(*)},ApplyLayout{typeof(*)}}) = _copy_ldiv_mul(L.A, arguments(ApplyLayout{typeof(*)}(), L.B)...)

@inline _copy_ldiv_ldiv(B, A₀) = A₀ \ B
@inline _copy_ldiv_ldiv(B, A₀, A₁...) = _copy_ldiv_ldiv(A₀ \ B, A₁...)
@inline copy(L::Ldiv{ApplyLayout{typeof(*)}}) = _copy_ldiv_ldiv(L.B, arguments(ApplyLayout{typeof(*)}(), L.A)...)
@inline copy(L::Ldiv{ApplyLayout{typeof(*)},<:AbstractLazyLayout}) = _copy_ldiv_ldiv(L.B, arguments(ApplyLayout{typeof(*)}(), L.A)...)
@inline copy(L::Ldiv{<:AbstractLazyLayout,<:AbstractLazyLayout}) = lazymaterialize(\, L.A, L.B)
@inline copy(L::Ldiv{<:AbstractLazyLayout}) = lazymaterialize(\, L.A, L.B)
@inline copy(L::Ldiv{<:Any,<:AbstractLazyLayout}) = lazymaterialize(\, L.A, L.B)

@inline copy(L::Ldiv{D,<:AbstractLazyLayout}) where D<:DiagonalLayout = copy(Ldiv{D,UnknownLayout}(L.A,L.B))

@inline copy(L::Rdiv{<:AbstractLazyLayout,<:AbstractLazyLayout}) = lazymaterialize(/, L.A, L.B)
@inline copy(L::Rdiv{<:AbstractLazyLayout}) = lazymaterialize(/, L.A, L.B)
@inline copy(L::Rdiv{<:Any,<:AbstractLazyLayout}) = lazymaterialize(/, L.A, L.B)


function copy(M::Mul{ApplyLayout{typeof(\)}})
    A,B = arguments(\, M.A)
    A \ (B * M.B)
end
copy(L::Mul{ApplyLayout{typeof(\)},<:AbstractLazyLayout}) = copy(Mul{ApplyLayout{typeof(\)},UnknownLayout}(L.A,L.B))

function copy(L::Ldiv{ApplyLayout{typeof(/)}})
    A,B = arguments(ApplyLayout{typeof(/)}(), L.A)
    B * (A \ L.B)
end
copy(L::Ldiv{ApplyLayout{typeof(/)},<:AbstractLazyLayout}) = copy(Ldiv{ApplyLayout{typeof(/)},UnknownLayout}(L.A,L.B))

###
# Diagonal
###

inv(D::Diagonal{T,<:LazyVector}) where T = Diagonal(inv.(D.diag))


###
# getindex
###

# \ is likely to be specialised
@propagate_inbounds getindex(Ai::InvMatrix{T}, ::Colon, j::Integer) where T = parent(Ai) \ [Zeros{T}(j-1); one(T); Zeros{T}(size(parent(Ai),1)-j)]
@propagate_inbounds getindex(Ai::PInvMatrix{T}, ::Colon, j::Integer) where T = parent(Ai) \ [Zeros{T}(j-1); one(T); Zeros{T}(size(parent(Ai),1)-j)]
getindex(Ai::SubArray{<:Any,2,<:InvMatrix}, ::Colon, j::Integer) = parent(Ai)[:, parentindices(Ai)[2][j]]

@propagate_inbounds getindex(A::PInvMatrix{T}, k::Integer, j::Integer) where T = A[:,j][k]
@propagate_inbounds getindex(A::InvMatrix{T}, k::Integer, j::Integer) where T = A[:,j][k]

getindex(L::ApplyMatrix{<:Any,typeof(\)}, ::Colon, j::Integer) where T = L.args[1] \ L.args[2][:,j]
getindex(L::ApplyMatrix{<:Any,typeof(\)}, k::Integer, j::Integer) where T = L[:,j][k]

getindex(L::ApplyMatrix{<:Any,typeof(/)}, k::Integer, ::Colon) where T = permutedims(L.args[2]) \ L.args[1][k,:]
getindex(L::ApplyMatrix{<:Any,typeof(/)}, k::Integer, j::Integer) where T = L[k,:][j]