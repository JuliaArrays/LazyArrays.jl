module LazyArraysBandedMatricesExt

using ArrayLayouts, BandedMatrices, LazyArrays
import ArrayLayouts: colsupport, rowsupport
import LazyArrays: sublayout, symmetriclayout, hermitianlayout
import Base: BroadcastStyle
import BandedMatrices: bandedbroadcaststyle, bandwidths, isbanded


abstract type AbstractLazyBandedLayout <: AbstractBandedLayout end
struct LazyBandedLayout <: AbstractLazyBandedLayout end
sublayout(::AbstractLazyBandedLayout, ::Type{<:NTuple{2,AbstractUnitRange}}) = LazyBandedLayout()
symmetriclayout(::AbstractLazyBandedLayout) = SymmetricLayout{LazyBandedLayout}()
hermitianlayout(::Type{<:Real}, ::AbstractLazyBandedLayout) = SymmetricLayout{LazyBandedLayout}()
hermitianlayout(::Type{<:Complex}, ::AbstractLazyBandedLayout) = HermitianLayout{LazyBandedLayout}()


bandedbroadcaststyle(::LazyArrayStyle) = LazyArrayStyle{2}()

BroadcastStyle(::LazyArrayStyle{1}, ::BandedStyle) = LazyArrayStyle{2}()
BroadcastStyle(::BandedStyle, ::LazyArrayStyle{1}) = LazyArrayStyle{2}()
BroadcastStyle(::LazyArrayStyle{2}, ::BandedStyle) = LazyArrayStyle{2}()
BroadcastStyle(::BandedStyle, ::LazyArrayStyle{2}) = LazyArrayStyle{2}()

BroadcastStyle(::LazyArrayStyle{N}, ::AbstractBlockStyle{N}) where N = LazyArrayStyle{N}()
BroadcastStyle(::AbstractBlockStyle{N}, ::LazyArrayStyle{N}) where N = LazyArrayStyle{N}()

bandedcolumns(::AbstractLazyLayout) = BandedColumns{LazyLayout}()
bandedcolumns(::DualLayout{<:AbstractLazyLayout}) = BandedColumns{LazyLayout}()

const StructuredLayoutTypes{Lay} = Union{SymmetricLayout{Lay}, HermitianLayout{Lay}, TriangularLayout{'L','N',Lay}, TriangularLayout{'U','N',Lay}, TriangularLayout{'L','U',Lay}, TriangularLayout{'U','U',Lay}}

const BandedLayouts = Union{AbstractBandedLayout, StructuredLayoutTypes{<:AbstractBandedLayout}, DualOrPaddedLayout}

BroadcastStyle(M::ApplyArrayBroadcastStyle{2}, ::BandedStyle) = M
BroadcastStyle(::BandedStyle, M::ApplyArrayBroadcastStyle{2}) = M

bandwidths(M::Applied{<:Any,typeof(*)}) = min.(_bnds(M), prodbandwidths(M.args...))

function bandwidths(L::ApplyMatrix{<:Any,typeof(\)})
    A,B = arguments(L)
    l,u = bandwidths(A)
    if l == u == 0
        bandwidths(B)
    elseif l == 0
        (bandwidth(B,1), size(L,2)-1)
    elseif u == 0
        (size(L,1)-1,bandwidth(B,2))
    else
        (size(L,1)-1 , size(L,2)-1)
    end
end

function bandwidths(L::ApplyMatrix{<:Any,typeof(inv)})
    A, = arguments(L)
    l,u = bandwidths(A)
    l == u == 0 && return (0,0)
    m,n = size(A)
    l == 0 && return (0,n-1)
    u == 0 && return (m-1,0)
    (m-1 , n-1)
end

function colsupport(::AbstractInvLayout{<:AbstractBandedLayout}, A, j)
    l,u = bandwidths(A)
    l == 0 && u == 0 && return first(j):last(j)
    m,_ = size(A)
    l == 0 && return 1:last(j)
    u == 0 && return first(j):m
    1:m
end

function rowsupport(::AbstractInvLayout{<:AbstractBandedLayout}, A, k)
    l,u = bandwidths(A)
    l == 0 && u == 0 && return first(k):last(k)
    _,n = size(A)
    l == 0 && return first(k):n
    u == 0 && return 1:last(k)
    1:n
end

isbanded(K::Kron{<:Any,2}) = all(isbanded, K.args)

function bandwidths(K::Kron{<:Any,2})
    A,B = K.args
    (size(B,1)*bandwidth(A,1) + max(0,size(B,1)-size(B,2))*size(A,1)   + bandwidth(B,1),
        size(B,2)*bandwidth(A,2) + max(0,size(B,2)-size(B,1))*size(A,2) + bandwidth(B,2))
end

const BandedMatrixTypes = (:AbstractBandedMatrix, :(AdjOrTrans{<:Any,<:AbstractBandedMatrix}),
                                    :(UpperOrLowerTriangular{<:Any, <:AbstractBandedMatrix}),
                                    :(Symmetric{<:Any, <:AbstractBandedMatrix}))

const OtherBandedMatrixTypes = (:Zeros, :Eye, :Diagonal, :(LinearAlgebra.SymTridiagonal))

for T1 in BandedMatrixTypes, T2 in BandedMatrixTypes
    @eval kron(A::$T1, B::$T2) = BandedMatrix(Kron(A,B))
end

for T1 in BandedMatrixTypes, T2 in OtherBandedMatrixTypes
    @eval kron(A::$T1, B::$T2) = BandedMatrix(Kron(A,B))
end

for T1 in OtherBandedMatrixTypes, T2 in BandedMatrixTypes
    @eval kron(A::$T1, B::$T2) = BandedMatrix(Kron(A,B))
end

###
# Columns as padded
# This is ommitted as it changes the behaviour of slicing B[:,4]
# it's activated in InfiniteLinearAlgebra
###

# sublayout(::AbstractBandedLayout, ::Type{<:Tuple{KR,Integer}}) where {KR<:AbstractUnitRange{Int}} = 
#     sublayout(PaddedLayout{UnknownLayout}(), Tuple{KR})
# sublayout(::AbstractBandedLayout, ::Type{<:Tuple{Integer,JR}}) where {JR<:AbstractUnitRange{Int}} = 
#     sublayout(PaddedLayout{UnknownLayout}(), Tuple{JR})

# function sub_paddeddata(::BandedColumns, S::SubArray{T,1,<:AbstractMatrix,<:Tuple{AbstractUnitRange{Int},Integer}}) where T
#     P = parent(S)
#     (kr,j) = parentindices(S)
#     data = bandeddata(P)
#     l,u = bandwidths(P)
#     Vcat(Zeros{T}(max(0,j-u-1)), view(data, (kr .- j .+ (u+1)) ∩ axes(data,1), j))
# end


end
