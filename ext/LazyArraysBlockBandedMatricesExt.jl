module LazyArraysBlockBandedMatricesExt


using BlockBandedMatrices, LazyArrays
using BlockBandedMatrices.BlockArrays
using BlockBandedMatrices.BandedMatrices
using LazyArrays.LinearAlgebra
using LazyArrays.ArrayLayouts
import LazyArrays: sublayout, symmetriclayout, hermitianlayout, transposelayout, conjlayout,
                PaddedLayout, PaddedColumns, AbstractPaddedLayout, PaddedRows, paddeddata,
                islazy_layout, arguments, call, applylayout, broadcastlayout, applybroadcaststyle,
                BroadcastMatrix, _broadcastarray2broadcasted, cache_layout, resizedata!, simplifiable,
                AbstractLazyLayout, LazyArrayStyle, LazyLayout, ApplyLayout, BroadcastLayout, AbstractInvLayout,
                _mul_args_colsupport, _mul_args_rowsupport, _mat_mul_arguments, LazyBandedLayout,
                CachedArray, _broadcast_sub_arguments, simplify, lazymaterialize, _mulbanded_copyto!
import BlockBandedMatrices: AbstractBlockBandedLayout, AbstractBandedBlockBandedLayout, blockbandwidths, subblockbandwidths,
                            bandedblockbandedbroadcaststyle, bandedblockbandedcolumns, BandedBlockBandedColumns, BandedBlockBandedRows,
                            BlockRange1, Block1, BlockIndexRange1, BlockBandedColumns, BlockBandedRows, BandedBlockBandedLayout
import BlockBandedMatrices.BlockArrays: BlockLayout, AbstractBlockLayout, BlockSlice, blockcolsupport, blockrowsupport
import BlockBandedMatrices.BandedMatrices: resize, BandedLayout, _copy_oftype
import Base: similar, copy, broadcasted
import ArrayLayouts: materialize!, MatMulVecAdd, sublayout, colsupport, rowsupport, copyto!_layout, mulreduce, inv_layout,
                        OnesLayout, AbstractFillLayout

abstract type AbstractLazyBlockBandedLayout <: AbstractBlockBandedLayout end
abstract type AbstractLazyBandedBlockBandedLayout <: AbstractBandedBlockBandedLayout end

struct LazyBlockBandedLayout <: AbstractLazyBlockBandedLayout end
struct LazyBandedBlockBandedLayout <: AbstractLazyBandedBlockBandedLayout end

const StructuredLayoutTypes{Lay} = Union{SymmetricLayout{Lay}, HermitianLayout{Lay}, TriangularLayout{'L','N',Lay}, TriangularLayout{'U','N',Lay}, TriangularLayout{'L','U',Lay}, TriangularLayout{'U','U',Lay}}
const BlockBandedLayouts = Union{AbstractBlockBandedLayout, BlockLayout{<:AbstractBandedLayout}, StructuredLayoutTypes{<:AbstractBlockBandedLayout}}
const BandedBlockBandedLayouts = Union{AbstractBandedBlockBandedLayout,DiagonalLayout{<:AbstractBlockLayout}, StructuredLayoutTypes{<:AbstractBandedBlockBandedLayout}}

const LazyBandedBlockBandedLayouts = Union{AbstractLazyBandedBlockBandedLayout,BandedBlockBandedColumns{<:AbstractLazyLayout}, BandedBlockBandedRows{<:AbstractLazyLayout}, StructuredLayoutTypes{<:AbstractLazyBandedBlockBandedLayout}}

transposelayout(::AbstractLazyBandedBlockBandedLayout) = LazyBandedBlockBandedLayout()
transposelayout(::AbstractLazyBlockBandedLayout) = LazyBlockBandedLayout()
conjlayout(::Type{<:Complex}, ::AbstractLazyBandedBlockBandedLayout) = LazyBandedBlockBandedLayout()
conjlayout(::Type{<:Complex}, ::AbstractLazyBlockBandedLayout) = LazyBlockBandedLayout()

symmetriclayout(::LazyBandedBlockBandedLayouts) = LazyBandedBlockBandedLayout()
hermitianlayout(_, ::LazyBandedBlockBandedLayouts) = LazyBandedBlockBandedLayout()

_block_last(b::Block) = b
_block_last(b::AbstractVector{<:Block}) = last(b)
function similar(Ml::MulAdd{<:BlockBandedLayouts,<:PaddedColumns}, ::Type{T}, _) where T
    A,x = Ml.A,Ml.B
    xf = paddeddata(x)
    ax1,ax2 = axes(A)
    N = findblock(ax2,length(xf))
    M = _block_last(blockcolsupport(A,N))
    isfinite(Integer(M)) || error("cannot multiply matrix with infinite block support")
    m = last(ax1[M]) # number of non-zero entries
    c = cache(Zeros{T}(length(ax1)))
    resizedata!(c, m)
    BlockedVector(c, (ax1,))
end

function materialize!(M::MatMulVecAdd{<:BlockBandedLayouts,<:PaddedColumns,<:PaddedColumns})
    α,A,x,β,y = M.α,M.A,M.B,M.β,M.C
    length(y) == size(A,1) || throw(DimensionMismatch())
    length(x) == size(A,2) || throw(DimensionMismatch())
    
    ỹ = paddeddata(y)

    if !blockisequal(axes(A,2), axes(x,1))
        x̃2 = paddeddata(x)
        muladd!(α, view(A, axes(ỹ,1), axes(x̃2,1)), x̃2, β, ỹ)
    else
        x̃ = paddeddata(x)
        muladd!(α, view(A, axes(ỹ,1), axes(x̃,1)), x̃, β, ỹ)
    end
    y
end

struct ApplyBlockBandedLayout{F} <: AbstractLazyBlockBandedLayout end
struct ApplyBandedBlockBandedLayout{F} <: AbstractLazyBandedBlockBandedLayout end
    

const ApplyBlockBandedLayouts{F} = Union{ApplyBlockBandedLayout{F},ApplyBandedBlockBandedLayout{F}}

LazyArrays._mul_arguments(::ApplyBlockBandedLayouts{F}, A) where F = LazyArrays._mul_arguments(ApplyLayout{F}(), A)
@inline islazy_layout(::ApplyBlockBandedLayouts) = Val(true)

# The following catches the arguments machinery to work for BlockRange
# see LazyArrays.jl/src/mul.jl

_mul_args_colsupport(a, kr::BlockRange) = blockcolsupport(a, kr)
_mul_args_rowsupport(a, kr::BlockRange) = blockrowsupport(a, kr)
_mul_args_colsupport(a, kr::Block) = blockcolsupport(a, kr)
_mul_args_rowsupport(a, kr::Block) = blockrowsupport(a, kr)
_mat_mul_arguments(args, (kr,jr)::Tuple{BlockSlice,BlockSlice}) = _mat_mul_arguments(args, (kr.block, jr.block))

arguments(::ApplyBlockBandedLayout{F}, A) where F = arguments(ApplyLayout{F}(), A)
arguments(::ApplyBandedBlockBandedLayout{F}, A) where F = arguments(ApplyLayout{F}(), A)

sublayout(::ApplyBlockBandedLayout{F}, A) where F = sublayout(ApplyLayout{F}(), A)
sublayout(::ApplyBandedBlockBandedLayout{F}, A) where F = sublayout(ApplyLayout{F}(), A)

sublayout(::ApplyBlockBandedLayout, ::Type{<:Tuple{<:BlockSlice{<:BlockRange1}, <:BlockSlice{<:BlockRange1}}}) = BlockBandedLayout()

sublayout(::ApplyBandedBlockBandedLayout{F}, ::Type{<:Tuple{BlockSlice{Block1},BlockSlice{Block1}}}) where F = BandedLayout()
sublayout(::ApplyBandedBlockBandedLayout{F}, ::Type{<:Tuple{BlockSlice{<:BlockIndexRange1},BlockSlice{<:BlockIndexRange1}}}) where F = BandedLayout()
sublayout(::ApplyBandedBlockBandedLayout{F}, ::Type{<:Tuple{BlockSlice{<:BlockIndexRange1},BlockSlice{Block1}}}) where F = BandedLayout()
sublayout(::ApplyBandedBlockBandedLayout{F}, ::Type{<:Tuple{BlockSlice{Block1},BlockSlice{<:BlockIndexRange1}}}) where F = BandedLayout()
sublayout(::ApplyBandedBlockBandedLayout{F}, ::Type{<:Tuple{BlockSlice{<:BlockRange1},BlockSlice{<:BlockRange1}}}) where F = BandedBlockBandedLayout()
sublayout(::ApplyBandedBlockBandedLayout{F}, ::Type{<:Tuple{BlockSlice{Block1},BlockSlice{<:BlockRange1}}}) where F = BandedBlockBandedLayout()
sublayout(::ApplyBandedBlockBandedLayout{F}, ::Type{<:Tuple{BlockSlice{<:BlockRange1},BlockSlice{Block1}}}) where F = BandedBlockBandedLayout()
sublayout(::ApplyBandedBlockBandedLayout{F}, ::Type{<:Tuple{BlockSlice{<:BlockRange1},BlockSlice{<:BlockIndexRange1}}}) where F = BandedBlockBandedLayout()
sublayout(::ApplyBandedBlockBandedLayout{F}, ::Type{<:Tuple{BlockSlice{<:BlockIndexRange1},BlockSlice{<:BlockRange1}}}) where F = BandedBlockBandedLayout()

applylayout(::Type{typeof(*)}, ::BlockBandedLayouts...) = ApplyBlockBandedLayout{typeof(*)}()
applylayout(::Type{typeof(*)}, ::BandedBlockBandedLayouts...) = ApplyBandedBlockBandedLayout{typeof(*)}()

applybroadcaststyle(::Type{<:AbstractMatrix}, ::ApplyBlockBandedLayout) = LazyArrayStyle{2}()
applybroadcaststyle(::Type{<:AbstractMatrix}, ::ApplyBandedBlockBandedLayout) = LazyArrayStyle{2}()

prodblockbandwidths(A) = blockbandwidths(A)
prodblockbandwidths(A...) = broadcast(+, blockbandwidths.(A)...)

prodsubblockbandwidths(A) = subblockbandwidths(A)
prodsubblockbandwidths(A...) = broadcast(+, subblockbandwidths.(A)...)

blockbandwidths(M::MulMatrix) = prodblockbandwidths(M.args...)
subblockbandwidths(M::MulMatrix) = prodsubblockbandwidths(M.args...)

###
# BroadcastMatrix
###

# TODO: Generalize
for op in (:+, :-)
    @eval begin
        blockbandwidths(M::BroadcastMatrix{<:Any,typeof($op)}) = broadcast(max, map(blockbandwidths,arguments(M))...)
        subblockbandwidths(M::BroadcastMatrix{<:Any,typeof($op)}) = broadcast(max, map(subblockbandwidths,arguments(M))...)
    end
end

for func in (:blockbandwidths, :subblockbandwidths)
    @eval begin
        $func(M::BroadcastMatrix{<:Any,typeof(*),<:Tuple{<:Number,<:AbstractMatrix}}) = $func(M.args[2])
        $func(M::BroadcastMatrix{<:Any,typeof(*),<:Tuple{<:AbstractMatrix,<:Number}}) = $func(M.args[1])
    end
end

struct BroadcastBlockBandedLayout{F} <: AbstractLazyBlockBandedLayout end
struct BroadcastBandedBlockBandedLayout{F} <: AbstractLazyBandedBlockBandedLayout end

const BroadcastBlockBandedLayouts{F} = Union{BroadcastBlockBandedLayout{F},BroadcastBandedBlockBandedLayout{F}}

blockbandwidths(B::BroadcastMatrix) = blockbandwidths(broadcasted(B))
subblockbandwidths(B::BroadcastMatrix) = subblockbandwidths(broadcasted(B))

# functions that satisfy f(0,0) == 0
for op in (:*, :-, :+, :sign, :abs)
    @eval begin
        broadcastlayout(::Type{typeof($op)}, ::BlockBandedLayouts) = BroadcastBlockBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::BandedBlockBandedLayouts) = BroadcastBandedBlockBandedLayout{typeof($op)}()
    end
end


for op in (:*, :/, :\, :+, :-)
    @eval begin
        broadcastlayout(::Type{typeof($op)}, ::BlockBandedLayouts, ::BlockBandedLayouts) = BroadcastBlockBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::BandedBlockBandedLayouts, ::BandedBlockBandedLayouts) = BroadcastBandedBlockBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::DiagonalLayout, ::AbstractBlockBandedLayout) = BroadcastBlockBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::AbstractBlockBandedLayout, ::DiagonalLayout) = BroadcastBlockBandedLayout{typeof($op)}()
    end
end

for op in (:*, :/)
    @eval begin
        broadcastlayout(::Type{typeof($op)}, ::BlockBandedLayouts, ::Any) = BroadcastBlockBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::BandedBlockBandedLayouts, ::Any) = BroadcastBandedBlockBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::BandedBlockBandedLayouts, ::DiagonalLayout) = BroadcastBandedBlockBandedLayout{typeof($op)}()
    end
end

for op in (:*, :\)
    @eval begin
        broadcastlayout(::Type{typeof($op)}, ::Any, ::BlockBandedLayouts) = BroadcastBlockBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::Any, ::BandedBlockBandedLayouts) = BroadcastBandedBlockBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::DiagonalLayout, ::BandedBlockBandedLayouts) = BroadcastBandedBlockBandedLayout{typeof($op)}()
    end
end

sublayout(LAY::BroadcastBlockBandedLayout, ::Type{<:Tuple{BlockSlice{<:BlockRange1},BlockSlice{<:BlockRange1}}}) = LAY
sublayout(LAY::BroadcastBandedBlockBandedLayout, ::Type{<:Tuple{BlockSlice{<:BlockRange1},BlockSlice{<:BlockRange1}}}) = LAY


_broadcastarray2broadcasted(::BroadcastBlockBandedLayouts{F}, A) where F = _broadcastarray2broadcasted(BroadcastLayout{F}(), A)
_broadcastarray2broadcasted(::BroadcastBlockBandedLayouts{F}, A::BroadcastArray) where F = _broadcastarray2broadcasted(BroadcastLayout{F}(), A)

function cache_layout(::BlockBandedLayouts, A::AbstractMatrix{T}) where T
    kr,jr = axes(A)
    CachedArray(BlockBandedMatrix{T}(undef, (kr[Block.(1:0)], jr[Block.(1:0)]), blockbandwidths(A)), A)
end

###
# copyto!
###

for op in (:+, :-, :*)
    @eval copyto!_layout(::AbstractBandedBlockBandedLayout, ::BroadcastBandedBlockBandedLayout{typeof($op)}, dest::AbstractMatrix, src::AbstractMatrix) =
            broadcast!($op, dest, map(_broadcast_BandedBlockBandedMatrix, arguments(src))...)
end


_mulbanded_BandedBlockBandedMatrix(A, _) = A
_mulbanded_BandedBlockBandedMatrix(A, ::NTuple{2,Int}) = BandedBlockBandedMatrix(A)
_mulbanded_BandedBlockBandedMatrix(A) = _mulbanded_BandedBlockBandedMatrix(A, size(A))

copyto!_layout(::AbstractBandedBlockBandedLayout, ::ApplyBandedBlockBandedLayout{typeof(*)}, dest::AbstractMatrix, src::AbstractMatrix) =
    _mulbanded_copyto!(dest, map(_mulbanded_BandedBlockBandedMatrix,arguments(src))...)

arguments(::BroadcastBandedBlockBandedLayout, V::SubArray) = _broadcast_sub_arguments(V)

sublayout(M::ApplyBlockBandedLayout{typeof(*)}, ::Type{<:Tuple{BlockSlice{<:BlockRange1},BlockSlice{<:BlockRange1}}}) = M
sublayout(M::ApplyBandedBlockBandedLayout{typeof(*)}, ::Type{<:Tuple{BlockSlice{<:BlockRange1},BlockSlice{<:BlockRange1}}}) = M

function resize(A::BlockSkylineMatrix{T}, ax::NTuple{2,AbstractUnitRange{Int}}) where T
    l,u = blockbandwidths(A)
    ret = BlockBandedMatrix{T}(undef, ax, (l,u))
    ret.data[1:length(A.data)] .= A.data
    ret
end

function resizedata!(laydat::BlockBandedColumns{<:AbstractColumnMajor}, layarr, B::AbstractMatrix, n::Integer, m::Integer)
    ν,μ = B.datasize
    n ≤ ν && m ≤ μ || resizedata!(laydat, layarr, B, findblock.(axes(B), (n,m))...)
end

function resizedata!(::BlockBandedColumns{<:AbstractColumnMajor}, _, B::AbstractMatrix{T}, N::Block{1}, M::Block{1}) where T<:Number
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


    if (ν,μ) ≠ (n,m)
        l,u = blockbandwidths(B.array)
        λ,ω = blockbandwidths(B.data)
        if Int(N) > blocksize(B.data,1) || Int(M) > blocksize(B.data,2)
            M̃ = 2*max(M,N+u)
            B.data = resize(olddata, (axes(B)[1][Block(1):min(M̃+λ,M_max)], axes(B)[2][Block(1):min(M̃,N_max)]))
        end
        if ν > 0 # upper-right
            KR = max(Block(1),M_old+1-ω):N_old
            JR = M_old+1:min(M,N_old+ω)
            if !isempty(KR) && !isempty(JR)
                copyto!(view(B.data, KR, JR), B.array[KR, JR])
            end
        end
        isempty(N_old+1:N) || isempty(M_old+1:M) || copyto!(view(B.data, N_old+1:N, M_old+1:M), B.array[N_old+1:N, M_old+1:M])
        if μ > 0
            KR = N_old+1:min(N,M_old+λ)
            JR = max(Block(1),N_old+1-λ):M_old
            if !isempty(KR) && !isempty(JR)
                view(B.data, KR, JR) .= B.array[KR, JR]
            end
        end
        B.datasize = (n,m)
    end

    B
end

bandedblockbandedbroadcaststyle(::LazyArrayStyle{2}) = LazyArrayStyle{2}()
bandedblockbandedcolumns(::LazyLayout) = BandedBlockBandedColumns{LazyLayout}()
bandedblockbandedcolumns(::ApplyLayout) = BandedBlockBandedColumns{LazyLayout}()
bandedblockbandedcolumns(::BroadcastLayout) = BandedBlockBandedColumns{LazyLayout}()

const LazyBlockBandedLayouts = Union{
                BlockBandedColumns{LazyLayout}, BandedBlockBandedColumns{LazyLayout},
                BlockBandedRows{LazyLayout},BandedBlockBandedRows{LazyLayout},
                BlockLayout{LazyLayout},
                BlockLayout{TridiagonalLayout{LazyLayout,LazyLayout,LazyLayout}}, BlockLayout{DiagonalLayout{LazyLayout}}, 
                BlockLayout{BidiagonalLayout{LazyLayout,LazyLayout}}, BlockLayout{SymTridiagonalLayout{LazyLayout,LazyLayout}},
                BlockLayout{LazyBandedLayout},
                AbstractLazyBlockBandedLayout, LazyBandedBlockBandedLayouts}



@inline islazy_layout(::LazyBlockBandedLayouts) = Val(true)

copy(M::Mul{<:LazyBlockBandedLayouts, <:LazyBlockBandedLayouts}) = simplify(M)
copy(M::Mul{<:LazyBlockBandedLayouts}) = simplify(M)
copy(M::Mul{<:Any, <:LazyBlockBandedLayouts}) = simplify(M)
copy(M::Mul{<:LazyBlockBandedLayouts, <:AbstractLazyLayout}) = simplify(M)
copy(M::Mul{<:AbstractLazyLayout, <:LazyBlockBandedLayouts}) = simplify(M)
copy(M::Mul{<:LazyBlockBandedLayouts, <:DiagonalLayout}) = simplify(M)
copy(M::Mul{<:DiagonalLayout, <:LazyBlockBandedLayouts}) = simplify(M)


copy(M::Mul{<:Union{ZerosLayout,DualLayout{ZerosLayout}}, <:LazyBlockBandedLayouts}) = copy(mulreduce(M))
copy(M::Mul{<:LazyBlockBandedLayouts, <:Union{ZerosLayout,DualLayout{ZerosLayout}}}) = copy(mulreduce(M))

simplifiable(::Mul{<:LazyBlockBandedLayouts, <:DiagonalLayout{<:OnesLayout}}) = Val(true)
simplifiable(::Mul{<:DiagonalLayout{<:OnesLayout}, <:LazyBlockBandedLayouts}) = Val(true)
copy(M::Mul{<:LazyBlockBandedLayouts, <:DiagonalLayout{<:OnesLayout}}) = _copy_oftype(M.A, eltype(M))
copy(M::Mul{<:DiagonalLayout{<:OnesLayout}, <:LazyBlockBandedLayouts}) = _copy_oftype(M.B, eltype(M))

copy(M::Mul{<:DiagonalLayout{<:AbstractFillLayout}, <:LazyBlockBandedLayouts}) = copy(mulreduce(M))
copy(M::Mul{<:LazyBlockBandedLayouts, <:DiagonalLayout{<:AbstractFillLayout}}) = copy(mulreduce(M))

copy(M::Mul{<:ApplyBlockBandedLayouts{typeof(*)},<:ApplyBlockBandedLayouts{typeof(*)}}) = simplify(M)
copy(M::Mul{<:ApplyBlockBandedLayouts{typeof(*)},<:LazyBlockBandedLayouts}) = simplify(M)
copy(M::Mul{<:LazyBlockBandedLayouts,<:ApplyBlockBandedLayouts{typeof(*)}}) = simplify(M)
copy(M::Mul{<:ApplyBlockBandedLayouts{typeof(*)},<:BroadcastBlockBandedLayouts}) = simplify(M)
copy(M::Mul{<:BroadcastBlockBandedLayouts,<:ApplyBlockBandedLayouts{typeof(*)}}) = simplify(M)
copy(M::Mul{BroadcastLayout{typeof(*)},<:ApplyBlockBandedLayouts{typeof(*)}}) = simplify(M)
copy(M::Mul{ApplyLayout{typeof(*)},<:LazyBlockBandedLayouts}) = simplify(M)
copy(M::Mul{<:LazyBlockBandedLayouts,ApplyLayout{typeof(*)}}) = simplify(M)
copy(M::Mul{ApplyLayout{typeof(*)},<:BroadcastBlockBandedLayouts}) = simplify(M)
copy(M::Mul{<:BroadcastBlockBandedLayouts,ApplyLayout{typeof(*)}}) = simplify(M)

copy(M::Mul{<:AbstractInvLayout, <:ApplyBlockBandedLayouts{typeof(*)}}) = simplify(M)
simplifiable(::Mul{<:AbstractInvLayout, <:LazyBlockBandedLayouts}) = Val(false)
copy(M::Mul{<:AbstractInvLayout, <:LazyBlockBandedLayouts}) = simplify(M)


copy(L::Ldiv{<:LazyBlockBandedLayouts, <:LazyBlockBandedLayouts}) = lazymaterialize(\, L.A, L.B)


copy(M::Mul{ApplyLayout{typeof(\)}, <:LazyBlockBandedLayouts}) = lazymaterialize(*, M.A, M.B)
copy(M::Mul{BroadcastLayout{typeof(*)}, <:LazyBlockBandedLayouts}) = lazymaterialize(*, M.A, M.B)

## padded copy
mulreduce(M::Mul{<:LazyBlockBandedLayouts, <:Union{AbstractPaddedLayout,AbstractStridedLayout}}) = MulAdd(M)
mulreduce(M::Mul{<:ApplyBlockBandedLayouts{F}, D}) where {F,D<:Union{AbstractPaddedLayout,AbstractStridedLayout}} = Mul{ApplyLayout{F},D}(M.A, M.B)
# need to overload copy due to above
copy(M::Mul{<:LazyBlockBandedLayouts, <:Union{AbstractPaddedLayout,AbstractStridedLayout}}) = copy(mulreduce(M))
simplifiable(::Mul{<:LazyBlockBandedLayouts, <:Union{AbstractPaddedLayout,AbstractStridedLayout}}) = Val(true)

inv_layout(::LazyBlockBandedLayouts, _, A) = ApplyArray(inv, A)

_broadcast_BandedBlockBandedMatrix(a::AbstractMatrix) = BandedBlockBandedMatrix(a)
_broadcast_BandedBlockBandedMatrix(a) = a



end
