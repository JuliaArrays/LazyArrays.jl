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

struct LdivApplyStyle <: ApplyStyle end


ldivapplystyle(_, _) = LdivApplyStyle()
ldivapplystyle(::LazyLayout, ::LazyLayout) = LazyArrayApplyStyle()
ldivapplystyle(::LazyLayout, _) = LazyArrayApplyStyle()
ldivapplystyle(_, ::LazyLayout) = LazyArrayApplyStyle()
ApplyStyle(::typeof(\), ::Type{A}, ::Type{B}) where {A<:AbstractArray,B<:AbstractArray} = 
    ldivapplystyle(MemoryLayout(A), MemoryLayout(B))


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
@inline copy(M::Applied{LdivApplyStyle}) = copy(Ldiv(M))
@inline copyto!(dest::AbstractArray, M::Applied{LdivApplyStyle}) = copyto!(dest, Ldiv(M))
@inline materialize!(M::Applied{LdivApplyStyle}) = materialize!(Ldiv(M))

@propagate_inbounds getindex(A::Applied{LazyArrayApplyStyle,typeof(\)}, kj...) = 
    materialize(Ldiv(A))[kj...]


###
# * layout
###

function copy(L::Ldiv{<:Any,ApplyLayout{typeof(*)}}) 
    args = arguments(L.B)
    apply(*, L.A \  first(args),  tail(args)...)
end