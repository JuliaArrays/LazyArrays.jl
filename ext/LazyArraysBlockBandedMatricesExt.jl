module LazyArraysBlockBandedMatricesExt

using BlockBandedMatrices
using LazyArrays
import LazyArrays: sublayout, symmetriclayout, hermitianlayout, transposelayout, conjlayout

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


end
