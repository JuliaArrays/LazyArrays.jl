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

ApplyStyle(::typeof(\), ::Type{A}, ::Type{B}) where {A<:AbstractArray,B<:AbstractArray} = LdivStyle()


axes(M::Applied{Style,typeof(\)}) where Style = ldivaxes(M.args...)
axes(M::Applied{Style,typeof(\)}, p::Int)  where Style = axes(M)[p]
size(M::Applied{Style,typeof(\)}) where Style = length.(axes(M))

@inline eltype(M::Applied{Style,typeof(\)}) where Style = eltype(Ldiv(M.args...))
@inline ndims(M::Applied{Style,typeof(\)}) where Style = ndims(last(M.args))

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


@propagate_inbounds getindex(A::PInvMatrix{T}, k::Int, j::Int) where T =
    (parent(A)\[Zeros(j-1); one(T); Zeros(size(A,2) - j)])[k]

@propagate_inbounds getindex(A::InvMatrix{T}, k::Int, j::Int) where T =
    (parent(A)\[Zeros(j-1); one(T); Zeros(size(A,2) - j)])[k]


abstract type AbstractInvLayout{L} <: MemoryLayout end
struct InvLayout{L} <: AbstractInvLayout{L} end
struct PInvLayout{L} <: AbstractInvLayout{L} end

applylayout(::Type{typeof(inv)}, ::A) where A = InvLayout{A}()
applylayout(::Type{typeof(pinv)}, ::A) where A = PInvLayout{A}()

copy(M::Mul{<:AbstractInvLayout}) = ArrayLayouts.ldiv(pinv(M.A), M.B)
Ldiv(A::Applied{<:Any,typeof(\)}) = Ldiv(A.args...)


similar(M::Applied{LdivStyle}, ::Type{T}) where T = similar(Ldiv(M), T)
@inline copy(M::Applied{LdivStyle}) = copy(Ldiv(M))
@inline copyto!(dest::AbstractArray, M::Applied{LdivStyle}) = copyto!(dest, Ldiv(M))
@inline materialize!(M::Applied{LdivStyle}) = materialize!(Ldiv(M))

@propagate_inbounds getindex(A::Applied{<:Any,typeof(\)}, kj...) = 
    materialize(Ldiv(A))[kj...]


###
# * layout
###

function copy(L::Ldiv{<:Any,ApplyLayout{typeof(*)}}) 
    args = arguments(L.B)
    apply(*, L.A \  first(args),  tail(args)...)
end