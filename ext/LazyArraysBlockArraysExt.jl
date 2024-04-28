module LazyArraysBlockArraysExt

using BlockArrays
using LazyArrays
using LazyArrays.ArrayLayouts
using LazyArrays.FillArrays
import LazyArrays: resizedata!, paddeddata, paddeddata_axes, arguments, call,
                    LazyArrayStyle, CachedVector, AbstractPaddedLayout, PaddedLayout, PaddedRows, PaddedColumns, BroadcastLayout,
                    AbstractCachedMatrix, AbstractCachedArray, setindex, applybroadcaststyle,
                    ApplyLayout
import ArrayLayouts: sub_materialize
import Base: getindex, BroadcastStyle, broadcasted, OneTo
import BlockArrays: AbstractBlockStyle, AbstractBlockedUnitRange, blockcolsupport, blockrowsupport, BlockSlice, BlockIndexRange

BlockArrays._broadcaststyle(S::LazyArrays.LazyArrayStyle{1}) = S

BroadcastStyle(::LazyArrayStyle{N}, ::AbstractBlockStyle{N}) where N = LazyArrayStyle{N}()
BroadcastStyle(::AbstractBlockStyle{N}, ::LazyArrayStyle{N}) where N = LazyArrayStyle{N}()


###
# Specialised multiplication for arrays padded for zeros
# needed for ∞-dimensional banded linear algebra
###



_makevec(data::AbstractVector) = data
_makevec(data::Number) = [data]

# make sure data is big enough for blocksize
function _block_paddeddata(C::CachedVector, data::AbstractVector, n)
    if n > length(data)
        resizedata!(C,n)
        data = paddeddata(C)
    end
    _makevec(data)
end

_block_paddeddata(C, data::Union{Number,AbstractVector}, n) = Vcat(data, Zeros{eltype(data)}(n-length(data)))
_block_paddeddata(C, data::Union{Number,AbstractMatrix}, n, m) = PaddedArray(data, n, m)

function resizedata!(P::PseudoBlockVector, n::Integer)
    ax = axes(P,1)
    N = findblock(ax,n)
    resizedata!(P.blocks, last(ax[N]))
    P
end

function paddeddata(P::PseudoBlockVector)
    C = P.blocks
    ax = axes(P,1)
    data = paddeddata(C)
    N = findblock(ax,max(length(data),1))
    n = last(ax[N])
    PseudoBlockVector(_block_paddeddata(C, data, n), (ax[Block(1):N],))
end

function paddeddata_axes((ax,)::Tuple{AbstractBlockedUnitRange}, A)
    data = A.args[2]
    N = findblock(ax,max(length(data),1))
    n = last(ax[N])
    PseudoBlockVector(_block_paddeddata(nothing, data, n), (ax[Block(1):N],))
end

function paddeddata(P::PseudoBlockMatrix)
    C = P.blocks
    ax,bx = axes(P)
    data = paddeddata(C)
    N = findblock(ax,max(size(data,1),1))
    M = findblock(bx,max(size(data,2),1))
    n,m = last(ax[N]),last(bx[M])
    PseudoBlockArray(_block_paddeddata(C, data, n, m), (ax[Block(1):N],bx[Block(1):M]))
end

blockcolsupport(::AbstractPaddedLayout, A, j) = Block.(OneTo(blocksize(paddeddata(A),1)))
blockrowsupport(::AbstractPaddedLayout, A, k) = Block.(OneTo(blocksize(paddeddata(A),2)))

function sub_materialize(::PaddedColumns, v::AbstractVector{T}, ax::Tuple{AbstractBlockedUnitRange}) where T
    dat = paddeddata(v)
    PseudoBlockVector(Vcat(sub_materialize(dat), Zeros{T}(length(v) - length(dat))), ax)
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


BroadcastStyle(M::Type{<:SubArray{<:Any,N,<:ApplyArray,I}}) where {N,I<:Tuple{BlockSlice{<:Any,<:BlockedUnitRange},Vararg{Any}}} = applybroadcaststyle(M, MemoryLayout(M))
BroadcastStyle(M::Type{<:SubArray{<:Any,N,<:ApplyArray,I}}) where {N,I<:Tuple{BlockSlice{<:Any,<:BlockedUnitRange},BlockSlice{<:Any,<:BlockedUnitRange},Vararg{Any}}} = applybroadcaststyle(M, MemoryLayout(M))
BroadcastStyle(M::Type{<:SubArray{<:Any,N,<:ApplyArray,I}}) where {N,I<:Tuple{Any,BlockSlice{<:Any,<:BlockedUnitRange},Vararg{Any}}} = applybroadcaststyle(M, MemoryLayout(M))

function getindex(A::ApplyMatrix{<:Any,typeof(*)}, kr::BlockRange{1}, jr::BlockRange{1})
    args = A.args
    kjr = intersect.(LazyArrays._mul_args_rows(kr, args...), LazyArrays._mul_args_cols(jr, reverse(args)...))
    *(map(getindex, args, (kr, kjr...), (kjr..., jr))...)
end

call(lay::BroadcastLayout, a::PseudoBlockArray) = call(lay, a.blocks)

resizedata!(lay1, lay2, B::AbstractMatrix, N::Block{2}) = resizedata!(lay1, lay2, B, Block.(N.n)...)

# Use memory laout for sub-blocks
@inline function getindex(A::AbstractCachedMatrix, K::Block{1}, J::Block{1})
    @boundscheck checkbounds(A, K, J)
    resizedata!(A, K, J)
    A.data[K, J]
end
@inline getindex(A::AbstractCachedMatrix, kr::Colon, jr::Block{1}) = ArrayLayouts.layout_getindex(A, kr, jr)
@inline getindex(A::AbstractCachedMatrix, kr::Block{1}, jr::Colon) = ArrayLayouts.layout_getindex(A, kr, jr)
@inline getindex(A::AbstractCachedMatrix, kr::Block{1}, jr::AbstractVector) = ArrayLayouts.layout_getindex(A, kr, jr)
@inline getindex(A::AbstractCachedArray{T,N}, kr::Block{1}, jrs...) where {T,N} = ArrayLayouts.layout_getindex(A, kr, jrs...)
@inline function getindex(A::AbstractCachedArray{T,N}, block::Block{N}) where {T,N}
    @boundscheck checkbounds(A, block)
    resizedata!(A, block)
    A.data[block]
end

@inline getindex(A::AbstractCachedMatrix, kr::AbstractVector, jr::Block) = ArrayLayouts.layout_getindex(A, kr, jr)
@inline getindex(A::AbstractCachedMatrix, kr::BlockRange{1}, jr::BlockRange{1}) = ArrayLayouts.layout_getindex(A, kr, jr)

###
# PseudoBlockArray apply
###

arguments(LAY::MemoryLayout, A::PseudoBlockArray) = arguments(LAY, A.blocks)

###
# work around bug in dat
###

LazyArrays._lazy_getindex(dat::PseudoBlockArray, kr::UnitRange) = view(dat.blocks,kr)
LazyArrays._lazy_getindex(dat::PseudoBlockArray, kr::OneTo) = view(dat.blocks,kr)


##
# support Inf Block ranges
broadcasted(::LazyArrayStyle{1}, ::Type{Block}, r::AbstractUnitRange) = Block(first(r)):Block(last(r))
broadcasted(::LazyArrayStyle{1}, ::Type{Int}, block_range::BlockRange{1}) = first(block_range.indices)
broadcasted(::LazyArrayStyle{0}, ::Type{Int}, block::Block{1}) = Int(block)


Base.in(K::Block, B::BroadcastVector{<:Block,Type{Block}}) = Int(K) in B.args[1]


###
# Concat
###

sub_materialize(lay::ApplyLayout{typeof(vcat)}, V::AbstractVector, ::Tuple{<:BlockedUnitRange}) = blockvcat(sub_materialize.(arguments(lay, V))...)
sub_materialize(lay::ApplyLayout{typeof(vcat)}, V::AbstractMatrix, ::Tuple{<:BlockedUnitRange,<:BlockedUnitRange}) = blockvcat(sub_materialize.(arguments(lay, V))...)
sub_materialize(lay::ApplyLayout{typeof(vcat)}, V::AbstractMatrix, ::Tuple{<:BlockedUnitRange,<:AbstractUnitRange}) = blockvcat(sub_materialize.(arguments(lay, V))...)

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


end
