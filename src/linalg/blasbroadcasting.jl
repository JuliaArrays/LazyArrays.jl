#########
# This file is to add support for lowering broadcast notation
#       y .= α .* Mul(A,x) .+ β .* y
# to
#       materialize!(MulAdd(α, A, x, β, y))
# which then becomes a blas call.
#########


struct ArrayMulArrayStyle{StyleA, StyleB, p, q} <: BroadcastStyle end

@inline copyto!(dest::AbstractArray, bc::Broadcasted{<:ArrayMulArrayStyle}) =
    _copyto!(MemoryLayout(dest), dest, bc)
# Use default broacasting in general
@inline _copyto!(_, dest, bc::Broadcasted) = copyto!(dest, Broadcasted{Nothing}(bc.f, bc.args, bc.axes))

const BArrayMulArray{styleA, styleB, p, q, T, V} =
    Broadcasted{ArrayMulArrayStyle{styleA,styleB,p,q}, <:Any, typeof(identity),
                <:Tuple{<:ArrayMulArray{styleA,styleB,p,q,T,V}}}
const BConstArrayMulArray{T, styleA, styleB, p, q} =
    Broadcasted{ArrayMulArrayStyle{styleA,styleB,p,q},
                    <:Any, typeof(*),
                    <:Tuple{T,<:ArrayMulArray{styleA,styleB,p,q,T,T}}}
const BArrayMulArrayPlusArray{T, styleA, styleB, p, q} =
    Broadcasted{ArrayMulArrayStyle{styleA,styleB,p,q},
                <:Any, typeof(+),
                <:Tuple{<:ArrayMulArray{styleA,styleB,p,q,T,T},<:AbstractArray{T,q}}}
const BArrayMulArrayPlusConstArray{T, styleA, styleB, p, q} =
    Broadcasted{ArrayMulArrayStyle{styleA,styleB,p,q},
                <:Any, typeof(+),
                <:Tuple{<:ArrayMulArray{styleA,styleB,p,q,T,T},
                Broadcasted{DefaultArrayStyle{q},<:Any,typeof(*),
                            <:Tuple{T,<:AbstractArray{T,q}}}}}
const BConstArrayMulArrayPlusArray{T, styleA, styleB, p, q} =
    Broadcasted{ArrayMulArrayStyle{styleA,styleB,p,q},
                <:Any, typeof(+),
                <:Tuple{Broadcasted{<:ArrayMulArrayStyle{styleA,styleB,p,q},
                                    <:Any, typeof(*),
                                    <:Tuple{T,<:ArrayMulArray{styleA,styleB,p,q,T,T}}},
                        <:AbstractArray{T,q}}}
const BConstArrayMulArrayPlusConstArray{T, styleA, styleB, p, q} =
    Broadcasted{ArrayMulArrayStyle{styleA,styleB,p,q},
                <:Any, typeof(+),
                <:Tuple{Broadcasted{<:ArrayMulArrayStyle{styleA,styleB,p,q},
                                    <:Any, typeof(*),
                                    <:Tuple{T,<:ArrayMulArray{styleA,styleB,p,q,T,T}}},
                        Broadcasted{DefaultArrayStyle{q},<:Any,typeof(*),<:Tuple{T,<:AbstractArray{T,q}}}}}


BroadcastStyle(::Type{<:ArrayMulArray{StyleA,StyleB,p,q}}) where {StyleA,StyleB,p,q} =
    ArrayMulArrayStyle{StyleA,StyleB,p,q}()
BroadcastStyle(M::ArrayMulArrayStyle, ::DefaultArrayStyle) = M
BroadcastStyle(::DefaultArrayStyle, M::ArrayMulArrayStyle) = M
similar(M::Broadcasted{<:ArrayMulArrayStyle}, ::Type{ElType}) where ElType =
    Array{ElType}(undef,size(M.args[1]))



broadcastable(M::ArrayMulArray) = M
instantiate(bc::Broadcasted{<:ArrayMulArrayStyle}) = bc


@inline function _copyto!(_, dest::AbstractArray{T}, M::ArrayMulArray) where T
    A,B = M.factors
    materialize!(MulAdd(one(T), A, B, zero(T), dest))
end

# Use copyto! for y .= Mul(A,b)
@inline function _copyto!(_, dest::AbstractArray, bc::BArrayMulArray)
    (M,) = bc.args
    copyto!(dest, M)
end

@inline function _copyto!(_, dest::AbstractArray{T}, bc::BConstArrayMulArray) where T
    α,M = bc.args
    A,B = M.factors
    materialize!(MulAdd(α, A, B, zero(T), dest))
end

@inline function _copyto!(_, dest::AbstractArray{T}, bc::BArrayMulArrayPlusArray) where T
    M,C = bc.args
    A,B = M.factors
    copyto!(dest, MulAdd(one(T), A, B, one(T), C))
end

@inline function _copyto!(_, dest::AbstractArray{T}, bc::BArrayMulArrayPlusConstArray) where T
    M,βc = bc.args
    β,C = βc.args
    A,B = M.factors
    copyto!(dest, MulAdd(one(T), A, B, β, C))
end

@inline function _copyto!(_, dest::AbstractArray{T}, bc::BConstArrayMulArrayPlusArray) where T
    αM,C = bc.args
    α,M = αM.args
    A,B = M.factors
    copyto!(dest, MulAdd(α, A, B, one(T), C))
end

@inline function _copyto!(_, dest::AbstractArray, bc::BConstArrayMulArrayPlusConstArray)
    αM,βc = bc.args
    α,M = αM.args
    A,B = M.factors
    β,C = βc.args
    copyto!(dest, MulAdd(α, A, B, β, C))
end
