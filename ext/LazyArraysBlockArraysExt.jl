module LazyArraysBlockArraysExt

using BlockArrays
using LazyArrays
using LazyArrays.ArrayLayouts
using LazyArrays.FillArrays
import LazyArrays: resizedata!, paddeddata, paddeddata_axes, arguments, call,
                    LazyArrayStyle, CachedVector, AbstractPaddedLayout, PaddedLayout, PaddedRows, PaddedColumns, BroadcastLayout,
                    AbstractCachedMatrix, AbstractCachedArray, setindex, applybroadcaststyle,
                    ApplyLayout, cache_layout
import ArrayLayouts: sub_materialize
import Base: getindex, BroadcastStyle, broadcasted, OneTo
import BlockArrays: AbstractBlockStyle, AbstractBlockedUnitRange, blockcolsupport, blockrowsupport, BlockSlice, BlockIndexRange, AbstractBlockLayout

BlockArrays._broadcaststyle(S::LazyArrays.LazyArrayStyle{1}) = S

BroadcastStyle(::LazyArrayStyle{N}, ::AbstractBlockStyle{N}) where N = LazyArrayStyle{N}()
BroadcastStyle(::AbstractBlockStyle{N}, ::LazyArrayStyle{N}) where N = LazyArrayStyle{N}()

BroadcastStyle(::Type{<:SubArray{<:Any,N,<:ApplyArray,I}}) where {N,I<:Tuple{BlockSlice{<:Any,<:Any,<:AbstractBlockedUnitRange},Vararg{Any}}} = LazyArrayStyle{N}()
BroadcastStyle(::Type{<:SubArray{<:Any,N,<:ApplyArray,I}}) where {N,I<:Tuple{BlockSlice{<:Any,<:Any,<:AbstractBlockedUnitRange},BlockSlice{<:Any,<:Any,<:AbstractBlockedUnitRange},Vararg{Any}}} = LazyArrayStyle{N}()
BroadcastStyle(::Type{<:SubArray{<:Any,N,<:ApplyArray,I}}) where {N,I<:Tuple{Any,BlockSlice{<:Any,<:Any,<:AbstractBlockedUnitRange},Vararg{Any}}} = LazyArrayStyle{N}()
###
# Specialised multiplication for arrays padded for zeros
# needed for ∞-dimensional banded linear algebra
###


# make sure data is big enough for blocksize
function _block_paddeddata(C::CachedVector, data::AbstractVector, n)
    if n > length(data)
        resizedata!(C,n)
        data = paddeddata(C)
    end
    data
end

_block_paddeddata(C, data::Union{Number,AbstractVector}, n) = Vcat(data, Zeros{eltype(data)}(n-length(data)))
_block_paddeddata(C, data::Union{Number,AbstractMatrix}, n, m) = PaddedArray(data, n, m)

function resizedata!(P::BlockedVector, n::Integer)
    ax = axes(P,1)
    N = findblock(ax,n)
    resizedata!(P.blocks, last(ax[N]))
    P
end

function paddeddata(P::BlockedVector)
    C = P.blocks
    ax = axes(P,1)
    data = paddeddata(C)
    N = findblock(ax,max(length(data),1))
    n = last(ax[N])
    BlockedVector(_block_paddeddata(C, data, n), (ax[Block(1):N],))
end

function paddeddata_axes((ax,)::Tuple{AbstractBlockedUnitRange}, A)
    data = A.args[2]
    N = findblock(ax,max(length(data),1))
    n = last(ax[N])
    BlockedVector(_block_paddeddata(nothing, data, n), (ax[Block(1):N],))
end

function paddeddata(P::BlockedMatrix)
    C = P.blocks
    ax,bx = axes(P)
    data = paddeddata(C)
    N = findblock(ax,max(size(data,1),1))
    M = findblock(bx,max(size(data,2),1))
    n,m = last(ax[N]),last(bx[M])
    BlockedArray(_block_paddeddata(C, data, n, m), (ax[Block(1):N],bx[Block(1):M]))
end

blockcolsupport(::AbstractPaddedLayout, A, j) = Block.(OneTo(blocksize(paddeddata(A),1)))
blockrowsupport(::AbstractPaddedLayout, A, k) = Block.(OneTo(blocksize(paddeddata(A),2)))

function sub_materialize(::PaddedColumns, v::AbstractVector{T}, ax::Tuple{AbstractBlockedUnitRange}) where T
    dat = paddeddata(v)
    BlockedVector(Vcat(sub_materialize(dat), Zeros{T}(length(v) - length(dat))), ax)
end

function sub_materialize(::AbstractPaddedLayout, V::AbstractMatrix{T}, ::Tuple{AbstractBlockedUnitRange,AbstractUnitRange}) where T
    dat = paddeddata(V)
    ApplyMatrix{T}(setindex, Zeros{T}(axes(V)), sub_materialize(dat), axes(dat)...)
end

function sub_materialize(::AbstractPaddedLayout, V::AbstractMatrix{T}, ::Tuple{AbstractBlockedUnitRange,AbstractBlockedUnitRange}) where T
    dat = paddeddata(V)
    ApplyMatrix{T}(setindex, Zeros{T}(axes(V)), sub_materialize(dat), axes(dat)...)
end

function sub_materialize(::AbstractPaddedLayout, V::AbstractMatrix{T}, ::Tuple{AbstractUnitRange,AbstractBlockedUnitRange}) where T
    dat = paddeddata(V)
    ApplyMatrix{T}(setindex, Zeros{T}(axes(V)), sub_materialize(dat), axes(dat)...)
end


BroadcastStyle(M::Type{<:SubArray{<:Any,N,<:ApplyArray,I}}) where {N,I<:Tuple{BlockSlice{<:Any,<:AbstractBlockedUnitRange},Vararg{Any}}} = applybroadcaststyle(M, MemoryLayout(M))
BroadcastStyle(M::Type{<:SubArray{<:Any,N,<:ApplyArray,I}}) where {N,I<:Tuple{BlockSlice{<:Any,<:AbstractBlockedUnitRange},BlockSlice{<:Any,<:AbstractBlockedUnitRange},Vararg{Any}}} = applybroadcaststyle(M, MemoryLayout(M))
BroadcastStyle(M::Type{<:SubArray{<:Any,N,<:ApplyArray,I}}) where {N,I<:Tuple{Any,BlockSlice{<:Any,<:AbstractBlockedUnitRange},Vararg{Any}}} = applybroadcaststyle(M, MemoryLayout(M))

function getindex(A::ApplyMatrix{<:Any,typeof(*)}, kr::BlockRange{1}, jr::BlockRange{1})
    args = A.args
    kjr = intersect.(LazyArrays._mul_args_rows(kr, args...), LazyArrays._mul_args_cols(jr, reverse(args)...))
    *(map(getindex, args, (kr, kjr...), (kjr..., jr))...)
end

call(lay::BroadcastLayout, a::BlockedArray) = call(lay, a.blocks)

resizedata!(lay1, lay2, B::AbstractMatrix, N::Block{2}) = resizedata!(lay1, lay2, B, Block.(N.n)...)

function resizedata!(lay1, lay2, B::AbstractMatrix{T}, N::Block{1}, M::Block{1}) where T<:Number
    (Int(N) ≤ 0 || Int(M) ≤ 0) && return B
    @boundscheck (N in blockaxes(B,1) && M in blockaxes(B,2)) || throw(ArgumentError("Cannot resize to ($N,$M) which is beyond size $(blocksize(B))"))


    N_max, M_max = Block.(blocksize(B))
    # increase size of array if necessary
    olddata = B.data
    ν,μ = B.datasize
    N_old = ν == 0 ? Block(0) : findblock(axes(B)[1], ν)
    M_old = μ == 0 ? Block(0) : findblock(axes(B)[2], μ)
    N,M = max(N_old,N),max(M_old,M)

    n,m = last.(getindex.(axes(B), (N,M)))
    resizedata!(B, n, m)

    B
end

# Use memory laout for sub-blocks
@inline getindex(A::AbstractCachedMatrix, kr::Colon, jr::Block{1}) = ArrayLayouts.layout_getindex(A, kr, jr)
@inline getindex(A::AbstractCachedMatrix, kr::Block{1}, jr::Colon) = ArrayLayouts.layout_getindex(A, kr, jr)
@inline getindex(A::AbstractCachedMatrix, kr::Block{1}, jr::AbstractVector) = ArrayLayouts.layout_getindex(A, kr, jr)
@inline getindex(A::AbstractCachedArray{T,N}, kr::Block{1}, jrs...) where {T,N} = ArrayLayouts.layout_getindex(A, kr, jrs...)
@inline function getindex(A::AbstractCachedArray{T,N}, block::Block{N}) where {T,N}
    @boundscheck checkbounds(A, block)
    resizedata!(A, block)
    A.data[getindex.(axes(A), Block.(block.n))...]
end

@inline getindex(A::AbstractCachedMatrix, kr::AbstractVector, jr::Block) = ArrayLayouts.layout_getindex(A, kr, jr)
@inline getindex(A::AbstractCachedMatrix, kr::BlockRange{1}, jr::BlockRange{1}) = ArrayLayouts.layout_getindex(A, kr, jr)

###
# BlockedArray apply
###

arguments(LAY::MemoryLayout, A::BlockedArray) = arguments(LAY, A.blocks)

###
# work around bug in dat
###

LazyArrays._lazy_getindex(dat::BlockedArray, kr::UnitRange) = view(dat.blocks,kr)
LazyArrays._lazy_getindex(dat::BlockedArray, kr::OneTo) = view(dat.blocks,kr)


##
# support Inf Block ranges
broadcasted(::LazyArrayStyle{1}, ::Type{Block}, r::AbstractUnitRange) = Block(first(r)):Block(last(r))
broadcasted(::LazyArrayStyle{1}, ::Type{Int}, block_range::BlockRange{1}) = first(block_range.indices)
broadcasted(::LazyArrayStyle{0}, ::Type{Int}, block::Block{1}) = Int(block)


Base.in(K::Block, B::BroadcastVector{<:Block,Type{Block}}) = Int(K) in B.args[1]


###
# Concat
###

LazyArrays._vcat_sub_arguments(lay::ApplyLayout{typeof(vcat)}, A, V, kr::BlockSlice{<:BlockRange{1}}) =
    arguments(lay, A)[Int.(kr.block)]


_split2blocks(KR) = ()
function _split2blocks(KR, ax::OneTo, C...)
    if isempty(KR)
        (Base.OneTo(0), _split2blocks(Block.(1:0), C...)...)
    elseif first(KR) ≠ Block(1)
        (Base.OneTo(0), _split2blocks((KR[1] - Block(1)):(KR[end] - Block(1)), C...)...)
    elseif length(KR) == 1
        (ax, _split2blocks(Block.(1:0), C...)...)
    else
        (ax, _split2blocks((KR[2]- Block(1)):(KR[end]-Block(1)), C...)...)
    end
end
function _split2blocks(KR, A, C...)
    M = blocklength(A)
    if Int(last(KR)) ≤ M
        (KR, _split2blocks(Block.(1:0), C...)...)
    else
        (KR[1]:Block(M), _split2blocks(Block(1):(last(KR)-Block(M)), C...)...)
    end
end

const Block1 = Block{1,Int}
const BlockRange1{R<:AbstractUnitRange{Int}} = BlockRange{1,Tuple{R}}
const BlockIndexRange1{R<:AbstractUnitRange{Int}} = BlockIndexRange{1,Tuple{R}}


function arguments(lay::ApplyLayout{typeof(vcat)}, V::SubArray{<:Any,1,<:Any,<:Tuple{BlockSlice{<:BlockRange1}}})
    kr, = parentindices(V)
    P = parent(V)
    a = arguments(lay, P)
    KR = _split2blocks(kr.block, axes.(a,1)...)
    filter(!isempty,getindex.(a, KR))
end


function arguments(lay::ApplyLayout{typeof(vcat)}, V::SubArray{<:Any,2,<:Any,<:Tuple{BlockSlice{<:BlockRange1},BlockSlice}})
    kr, jr = parentindices(V)
    P = parent(V)
    a = arguments(lay, P)
    KR = _split2blocks(kr.block, axes.(a,1)...)
    filter(!isempty,getindex.(a, KR, Ref(jr.block)))
end

function arguments(lay::ApplyLayout{typeof(hcat)}, V::SubArray{<:Any,2,<:Any,<:Tuple{BlockSlice,BlockSlice{<:BlockRange1}}})
    kr, jr = parentindices(V)
    P = parent(V)
    a = arguments(lay, P)
    JR = _split2blocks(jr.block, axes.(a,2)...)
    filter(!isempty,getindex.(a, Ref(kr.block), JR))
end

function arguments(lay::ApplyLayout{typeof(hcat)}, V::SubArray{<:Any,2,<:Any,<:Tuple{BlockSlice{<:BlockRange1},BlockSlice{<:Block1}}})
    kr, jr = parentindices(V)
    J = jr.block
    P = parent(V)
    arguments(lay, view(P,kr,J:J))
end



end
