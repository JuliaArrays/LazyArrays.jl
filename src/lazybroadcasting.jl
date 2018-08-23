struct LazyArrayStyle{N} <: AbstractArrayStyle{N} end
LazyArrayStyle(::Val{N}) where N = LazyArrayStyle{N}()
LazyArrayStyle{M}(::Val{N}) where {N,M} = LazyArrayStyle{N}()


struct BroadcastArray{T, N, BRD<:Broadcasted} <: AbstractArray{T, N}
    broadcasted::BRD
end

BroadcastArray{T,N}(bc::BRD) where {T,N,BRD<:Broadcasted} = BroadcastArray{T,N,BRD}(bc)
BroadcastArray{T}(bc::Broadcasted{<:Union{Nothing,BroadcastStyle},<:Tuple{Vararg{Any,N}},<:Any,<:Tuple}) where {T,N} =
    BroadcastArray{T,N}(bc)
BroadcastArray(bc::Broadcasted) =
    BroadcastArray{combine_eltypes(bc.f, bc.args)}(bc)
BroadcastArray(b::Broadcasted{<:Union{Nothing,BroadcastStyle},Nothing,<:Any,<:Tuple}) =
    BroadcastArray(instantiate(b))
BroadcastArray(f, A, As...) = BroadcastArray(broadcasted(f, A, As...))

axes(A::BroadcastArray) = axes(A.broadcasted)
size(A::BroadcastArray) = map(length, axes(A))

IndexStyle(::BroadcastArray{<:Any,1}) = IndexLinear()

@propagate_inbounds getindex(A::BroadcastArray, kj::Int...) = A.broadcasted[kj...]

copy(bc::Broadcasted{<:LazyArrayStyle}) = BroadcastArray(bc)



## scalar-range broadcast operations ##
# Ranges already support smart broadcasting
for op in (+, -, big)
    @eval begin
        broadcasted(::LazyArrayStyle{1}, ::typeof($op), r::AbstractRange) =
            broadcast(DefaultArrayStyle{1}(), $op, r)
    end
end

for op in (-, +, *, /)
    @eval broadcasted(::LazyArrayStyle{1}, ::typeof($op), r::AbstractFill, x::Real) =
        broadcast(DefaultArrayStyle{1}(), $op, r, x)
end

for op in (-, +, *, \)
    @eval broadcasted(::LazyArrayStyle{1}, ::typeof($op), x::Real, r::AbstractFill) =
        broadcast(DefaultArrayStyle{1}(), $op, x, r)
end


broadcasted(::LazyArrayStyle{N}, op, r::AbstractFill{T,N}) where {T,N} =
    broadcast(DefaultArrayStyle{N}(), op, r)
broadcasted(::LazyArrayStyle{N}, op, r::AbstractFill{T,N}, x::Number) where {T,N} = broadcast(DefaultArrayStyle{N}(), op, r, x)
broadcasted(::LazyArrayStyle{N}, op, x::Number, r::AbstractFill{T,N}) where {T,N} =
    broadcast(DefaultArrayStyle{N}(), op, x, r)
broadcasted(::LazyArrayStyle{N}, op, r1::AbstractFill{T,N}, r2::AbstractFill{V,N}) where {T,V,N} =
    broadcast(DefaultArrayStyle{N}(), op, r1, r2)
