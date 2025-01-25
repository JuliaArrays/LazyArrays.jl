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




for op in (:inv, :pinv)
    @eval begin
        @inline applied_size(::typeof($op), a) = reverse(size(a))
        @inline applied_axes(::typeof($op), a) = reverse(axes(a))
        @inline applied_eltype(::typeof($op), a) = Base.promote_op(inv, eltype(a))
        @inline applied_ndims(::typeof($op), a) = 2
    end
end

# Use ArrayLayouts.ldiv instead of \
struct LdivStyle <: ApplyStyle end
struct RdivStyle <: ApplyStyle end

@inline ApplyStyle(::typeof(\), ::Type{A}, ::Type{B}) where {A<:AbstractArray,B<:AbstractArray} = LdivStyle()
@inline ApplyStyle(::typeof(/), ::Type{A}, ::Type{B}) where {A<:AbstractArray,B<:AbstractArray} = RdivStyle()


@inline applied_axes(::typeof(\), args...) = ldivaxes(args...)
@inline applied_size(::typeof(\), args...) = length.(applied_axes(\, args...))
@inline applied_eltype(::typeof(\), args...) = eltype(Ldiv(args...))
@inline applied_ndims(::typeof(\), args...) = ndims(last(args))


@inline applied_axes(::typeof(/), args...) = axes(Rdiv(args...))
@inline applied_size(::typeof(/), args...) = length.(applied_axes(/, args...))
@inline applied_eltype(::typeof(/), args...) = eltype(Rdiv(args...))
@inline applied_ndims(::typeof(/), args...) = ndims(first(args))


check_applied_axes(::typeof(\), args...) = check_ldiv_axes(args...)

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



abstract type AbstractInvLayout{L} <: AbstractLazyLayout end
struct InvLayout{L} <: AbstractInvLayout{L} end
struct PInvLayout{L} <: AbstractInvLayout{L} end

applylayout(::Type{typeof(inv)}, ::A) where A = InvLayout{A}()
applylayout(::Type{typeof(pinv)}, ::A) where A = PInvLayout{A}()

# Can always  simplify by lowering to \
simplifiable(::Mul{<:AbstractInvLayout}) = Val(true)

copy(M::Mul{<:AbstractInvLayout}) = ArrayLayouts.ldiv(pinv(M.A), M.B)
copy(M::Mul{<:AbstractInvLayout, <:AbstractLazyLayout}) = ArrayLayouts.ldiv(pinv(M.A), M.B)
@inline copy(M::Mul{<:AbstractInvLayout, <:DiagonalLayout{<:AbstractFillLayout}}) = copy(mulreduce(M))
@inline copy(M::Mul{<:AbstractInvLayout, ApplyLayout{typeof(*)}}) = simplify(M)
copy(L::Ldiv{<:AbstractInvLayout}) = pinv(L.A) * L.B
copy(L::Ldiv{<:AbstractInvLayout, <:AbstractLazyLayout}) = pinv(L.A) * L.B
copy(L::Ldiv{<:AbstractInvLayout, <:AbstractInvLayout}) = pinv(L.A) * L.B
copy(L::Ldiv{<:AbstractInvLayout, ApplyLayout{typeof(*)}}) = pinv(L.A) * L.B
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
    simplifiable(*, B₀, B₁...) isa Val{true} && return A \ *(B₀, B₁...)
    AB₀ = A \  B₀
    simplifiable(*, AB₀,  B₁...) isa Val{true} && return *(AB₀, B₁...)
    lazymaterialize(*, AB₀, B₁...)
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
@inline copy(L::Ldiv{D,<:AbstractLazyLayout}) where {D<:DiagonalLayout{<:AbstractFillLayout}} = copy(Ldiv{D, UnknownLayout}(L.A, L.B))

@inline copy(L::Rdiv{<:AbstractLazyLayout,<:AbstractLazyLayout}) = lazymaterialize(/, L.A, L.B)
@inline copy(L::Rdiv{<:AbstractLazyLayout}) = lazymaterialize(/, L.A, L.B)
@inline copy(L::Rdiv{<:Any,<:AbstractLazyLayout}) = lazymaterialize(/, L.A, L.B)


@inline simplifiable(L::Ldiv) = _not(_or(islazy(L.A), islazy(L.B)))
@inline simplifiable(L::Ldiv{<:Any,ApplyLayout{typeof(*)}}) = simplifiable(\, L.A, first(arguments(*, L.B)))
@inline simplifiable(::typeof(\), a, b) = simplifiable(Ldiv(a,b))

simplifiable(M::Mul{ApplyLayout{typeof(\)}}) = simplifiable(*, last(arguments(\, M.A)), M.B)
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
struct InvColumnLayout <: AbstractLazyLayout end


sublayout(::AbstractInvLayout, I::Type{<:Tuple{Slice, Int}}) = InvColumnLayout()

function sub_materialize(::InvColumnLayout, v::AbstractVector, _)
    _,j = parentindices(v)
    Ai = parent(v)
    T = eltype(v)
    parent(Ai) \ [Zeros{T}(j-1); one(T); Zeros{T}(size(parent(Ai),1)-j)]
end

@propagate_inbounds getindex(A::PInvMatrix{T}, k::Integer, j::Integer) where T = A[:,j][k]
@propagate_inbounds getindex(A::InvMatrix{T}, k::Integer, j::Integer) where T = A[:,j][k]

getindex(L::ApplyMatrix{<:Any,typeof(\)}, ::Colon, j::Integer) = L.args[1] \ L.args[2][:,j]
getindex(L::ApplyMatrix{<:Any,typeof(\)}, k::Integer, j::Integer) = L[:,j][k]

getindex(L::ApplyMatrix{<:Any,typeof(/)}, k::Integer, ::Colon) = permutedims(L.args[2]) \ L.args[1][k,:]
getindex(L::ApplyMatrix{<:Any,typeof(/)}, k::Integer, j::Integer) = L[k,:][j]


inv_layout(::LazyLayouts, _, A) = ApplyArray(inv, A)

function colsupport(lay::AbstractInvLayout{TriLay}, A, j) where {S,TriLay<:TriangularLayout{S}}
    isempty(j) && return 1:0
    B, = arguments(lay, A)
    if S == 'U'
        return firstindex(B, 2):(maximum(j) - firstindex(B, 2) + 1)
    else # S == 'L' 
        return minimum(j):size(B, 2)
    end 
end 
function rowsupport(lay::AbstractInvLayout{TriLay}, A, k) where {S,TriLay<:TriangularLayout{S}}
    isempty(k) && return 1:0
    B, = arguments(lay, A) 
    if S == 'U' 
        return minimum(k):size(B, 1) 
    else # S == 'L'
        return firstindex(B, 1):(maximum(k) - firstindex(B, 1) + 1)
    end 
end