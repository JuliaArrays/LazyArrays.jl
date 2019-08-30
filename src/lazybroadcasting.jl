struct LazyArrayStyle{N} <: AbstractArrayStyle{N} end
LazyArrayStyle(::Val{N}) where N = LazyArrayStyle{N}()
LazyArrayStyle{M}(::Val{N}) where {N,M} = LazyArrayStyle{N}()


struct BroadcastArray{T, N, F, Args} <: LazyArray{T, N}
    f::F
    args::Args
end

const BroadcastVector{T,F,Args} = BroadcastArray{T,1,F,Args}
const BroadcastMatrix{T,F,Args} = BroadcastArray{T,2,F,Args}

LazyArray(bc::Broadcasted) = BroadcastArray(bc)

BroadcastArray{T,N,F,Args}(bc::Broadcasted) where {T,N,F,Args} = BroadcastArray{T,N,F,Args}(bc.f,bc.args)
BroadcastArray{T,N}(bc::Broadcasted{Style,Axes,F,Args}) where {T,N,Style,Axes,F,Args} = BroadcastArray{T,N,F,Args}(bc.f,bc.args)
BroadcastArray{T}(bc::Broadcasted{<:Union{Nothing,BroadcastStyle},<:Tuple{Vararg{Any,N}},<:Any,<:Tuple}) where {T,N} =
    BroadcastArray{T,N}(bc)

_broadcast2broadcastarray(a, b...) = tuple(a, b...)
_broadcast2broadcastarray(a::Broadcasted, b...) = tuple(BroadcastArray(a), b...)

_BroadcastArray(bc::Broadcasted) = BroadcastArray{combine_eltypes(bc.f, bc.args)}(bc)
BroadcastArray(bc::Broadcasted{S}) where S =
    _BroadcastArray(instantiate(Broadcasted{S}(bc.f, _broadcast2broadcastarray(bc.args...))))
BroadcastArray(b::BroadcastArray) = b
BroadcastArray(f, A, As...) = BroadcastArray(broadcasted(f, A, As...))

Broadcasted(A::BroadcastArray) = instantiate(broadcasted(A.f, A.args...))


axes(A::BroadcastArray) = axes(Broadcasted(A))
size(A::BroadcastArray) = map(length, axes(A))

IndexStyle(::BroadcastArray{<:Any,1}) = IndexLinear()

@propagate_inbounds getindex(A::BroadcastArray, kj::Int...) = Broadcasted(A)[kj...]


@propagate_inbounds _broadcast_getindex_range(A::Union{Ref,AbstractArray{<:Any,0},Number}, I) = A[] # Scalar-likes can just ignore all indices
# Everything else falls back to dynamically dropping broadcasted indices based upon its axes
@propagate_inbounds _broadcast_getindex_range(A, I) = A[I]

getindex(B::BroadcastArray{<:Any,1}, kr::AbstractVector{<:Integer}) =
    BroadcastArray(Broadcasted(B).f, map(a -> _broadcast_getindex_range(a,kr), Broadcasted(B).args)...)

copy(bc::Broadcasted{<:LazyArrayStyle}) = BroadcastArray(bc)

# Replacement for #18.
# Could extend this to other similar reductions in Base... or apply at lower level? 
# for (fname, op) in [(:sum, :add_sum), (:prod, :mul_prod),
#                     (:maximum, :max), (:minimum, :min),
#                     (:all, :&),       (:any, :|)]
function Base._sum(f, A::BroadcastArray, ::Colon)
    bc = Broadcasted(A)
    T = Broadcast.combine_eltypes(f ∘ bc.f, bc.args) 
    out = zero(T)
    @simd for I in eachindex(bc)
        @inbounds out += f(bc[I])
    end
    out
end
function Base._prod(f, A::BroadcastArray, ::Colon)
    bc = Broadcasted(A)
    T = Broadcast.combine_eltypes(f ∘ bc.f, bc.args) 
    out = one(T)
    @simd for I in eachindex(bc)
        @inbounds out *= f(bc[I])
    end
    out
end


BroadcastStyle(::Type{<:BroadcastArray{<:Any,N}}) where N = LazyArrayStyle{N}()
BroadcastStyle(L::LazyArrayStyle{N}, ::StaticArrayStyle{N}) where N = L
BroadcastStyle(::StaticArrayStyle{N}, L::LazyArrayStyle{N})  where N = L

"""
    BroadcastLayout(f, layouts)

is returned by `MemoryLayout(A)` if a matrix `A` is a `BroadcastArray`.
`f` is a function that broadcast operation is applied and `layouts` is
a tuple of `MemoryLayout` of the broadcasted arguments.
"""
struct BroadcastLayout{F, LAY} <: MemoryLayout end

tuple_type_memorylayouts(::Type{I}) where I<:Tuple = Tuple{typeof.(MemoryLayout.(I.parameters))...}
tuple_type_memorylayouts(::Type{Tuple{A}}) where {A} = Tuple{typeof(MemoryLayout(A))}
tuple_type_memorylayouts(::Type{Tuple{A,B}}) where {A,B} = Tuple{typeof(MemoryLayout(A)),typeof(MemoryLayout(B))}
tuple_type_memorylayouts(::Type{Tuple{A,B,C}}) where {A,B,C} = Tuple{typeof(MemoryLayout(A)),typeof(MemoryLayout(B)),typeof(MemoryLayout(C))}

MemoryLayout(::Type{BroadcastArray{T,N,F,Args}}) where {T,N,F,Args} = 
    BroadcastLayout{F,tuple_type_memorylayouts(Args)}()

## scalar-range broadcast operations ##
# Ranges already support smart broadcasting
for op in (+, -, big)
    @eval begin
        broadcasted(::LazyArrayStyle{1}, ::typeof($op), r::AbstractRange) =
            broadcast(DefaultArrayStyle{1}(), $op, r)
    end
end

for op in (-, +, *, /)
    @eval broadcasted(::LazyArrayStyle{1}, ::typeof($op), r::AbstractRange, x::Real) =
        broadcast(DefaultArrayStyle{1}(), $op, r, x)
end

for op in (-, +, *, \)
    @eval broadcasted(::LazyArrayStyle{1}, ::typeof($op), x::Real, r::AbstractRange) =
        broadcast(DefaultArrayStyle{1}(), $op, x, r)
end


broadcasted(::LazyArrayStyle{N}, op, r::AbstractFill{T,N}) where {T,N} =
    broadcast(DefaultArrayStyle{N}(), op, r)
broadcasted(::LazyArrayStyle{N}, op, r::AbstractFill{T,N}, x::Number) where {T,N} =
    broadcast(DefaultArrayStyle{N}(), op, r, x)
broadcasted(::LazyArrayStyle{N}, op, x::Number, r::AbstractFill{T,N}) where {T,N} =
    broadcast(DefaultArrayStyle{N}(), op, x, r)
broadcasted(::LazyArrayStyle{N}, op, r::AbstractFill{T,N}, x::Ref) where {T,N} =
    broadcast(DefaultArrayStyle{N}(), op, r, x)
broadcasted(::LazyArrayStyle{N}, op, x::Ref, r::AbstractFill{T,N}) where {T,N} =
    broadcast(DefaultArrayStyle{N}(), op, x, r)    
broadcasted(::LazyArrayStyle{N}, op, r1::AbstractFill{T,N}, r2::AbstractFill{V,N}) where {T,V,N} =
    broadcast(DefaultArrayStyle{N}(), op, r1, r2)

broadcasted(::LazyArrayStyle{N}, ::typeof(*), a::Zeros{T,N}, b::Zeros{V,N}) where {T,V,N} =
    broadcast(DefaultArrayStyle{N}(), *, a, b)
broadcasted(::LazyArrayStyle{N}, ::typeof(*), a::AbstractArray{T,N}, b::Zeros{V,N}) where {T,V,N} =
    broadcast(DefaultArrayStyle{N}(), *, a, b)
broadcasted(::LazyArrayStyle{N}, ::typeof(*), a::Zeros{T,N}, b::AbstractArray{V,N}) where {T,V,N} =
    broadcast(DefaultArrayStyle{N}(), *, a, b)
