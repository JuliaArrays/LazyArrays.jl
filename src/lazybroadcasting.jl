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


@propagate_inbounds getindex(A::BroadcastArray, kj...) = A.broadcasted[kj...]

copy(bc::Broadcasted{<:LazyArrayStyle}) = BroadcastArray(bc)



## scalar-range broadcast operations ##
# Ranges already support smart broadcasting
for op in (+, -, big)
    @eval begin
        broadcasted(::LazyArrayStyle{1}, ::typeof($op), r::AbstractRange) = broadcast(DefaultArrayStyle{1}(), $op, r)
        broadcasted(::LazyArrayStyle{1}, ::typeof($op), r1::AbstractRange, r2::AbstractRange) = broadcast(DefaultArrayStyle{1}(), $op, r1, r2)
    end
end

for op in (-, +, *, /)
    @eval broadcasted(::LazyArrayStyle{1}, ::typeof($op), r::AbstractRange, x::Real) = broadcast(DefaultArrayStyle{1}(), $op, r, x)
end

for op in (-, +, *, \)
    @eval broadcasted(::LazyArrayStyle{1}, ::typeof($op), x::Real, r::AbstractRange) = broadcast(DefaultArrayStyle{1}(), $op, x, r)
end
