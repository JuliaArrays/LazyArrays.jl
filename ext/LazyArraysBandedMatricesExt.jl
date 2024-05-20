module LazyArraysBandedMatricesExt

using BandedMatrices, LazyArrays, LinearAlgebra
using LazyArrays.ArrayLayouts, LazyArrays.FillArrays, LazyArrays.LazyArrays
import ArrayLayouts: colsupport, rowsupport, materialize!, MatMulVecAdd, MatMulMatAdd, DenseColumnMajor,
                    OnesLayout, AbstractFillLayout, mulreduce, _inv, _fill_lmul!, _copyto!, _copy_oftype
import LazyArrays: sublayout, symmetriclayout, hermitianlayout, applylayout, cachedlayout, transposelayout,
                   LazyArrayStyle, ApplyArrayBroadcastStyle, AbstractInvLayout, AbstractLazyLayout,
                   AbstractPaddedLayout, PaddedLayout, PaddedRows, PaddedColumns, CachedArray, CachedMatrix, LazyLayout, BroadcastLayout, ApplyLayout,
                   paddeddata, resizedata!, broadcastlayout, _broadcastarray2broadcasted, _broadcast_sub_arguments,
                   arguments, call, applybroadcaststyle, simplify, simplifiable, islazy_layout, lazymaterialize, _broadcast_mul_mul
import Base: BroadcastStyle, similar, copy, broadcasted, getindex, OneTo, oneto, tail
import BandedMatrices: bandedbroadcaststyle, bandwidths, isbanded, bandedcolumns, bandeddata, BandedStyle,
                        AbstractBandedLayout, AbstractBandedMatrix, BandedColumns, BandedRows, BandedSubBandedMatrix, 
                        _bnds, prodbandwidths, banded_rowsupport, banded_colsupport, _BandedMatrix, _banded_broadcast!,
                        resize
import LinearAlgebra: AdjOrTrans, UpperOrLowerTriangular, kron

abstract type AbstractLazyBandedLayout <: AbstractBandedLayout end
struct LazyBandedLayout <: AbstractLazyBandedLayout end

symmetriclayout(::AbstractLazyBandedLayout) = SymmetricLayout{LazyBandedLayout}()
hermitianlayout(::Type{<:Real}, ::AbstractLazyBandedLayout) = SymmetricLayout{LazyBandedLayout}()
hermitianlayout(::Type{<:Complex}, ::AbstractLazyBandedLayout) = HermitianLayout{LazyBandedLayout}()


bandedbroadcaststyle(::LazyArrayStyle) = LazyArrayStyle{2}()

BroadcastStyle(::LazyArrayStyle{1}, ::BandedStyle) = LazyArrayStyle{2}()
BroadcastStyle(::BandedStyle, ::LazyArrayStyle{1}) = LazyArrayStyle{2}()
BroadcastStyle(::LazyArrayStyle{2}, ::BandedStyle) = LazyArrayStyle{2}()
BroadcastStyle(::BandedStyle, ::LazyArrayStyle{2}) = LazyArrayStyle{2}()

bandedcolumns(::AbstractLazyLayout) = BandedColumns{LazyLayout}()
bandedcolumns(::DualLayout{<:AbstractLazyLayout}) = BandedColumns{LazyLayout}()

const StructuredLayoutTypes{Lay} = Union{SymmetricLayout{Lay}, HermitianLayout{Lay}, TriangularLayout{'L','N',Lay}, TriangularLayout{'U','N',Lay}, TriangularLayout{'L','U',Lay}, TriangularLayout{'U','U',Lay}}

const BandedLayouts = Union{AbstractBandedLayout, StructuredLayoutTypes{<:AbstractBandedLayout}}

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

function similar(M::MulAdd{<:BandedLayouts,<:AbstractPaddedLayout}, ::Type{T}, axes::Tuple{Any}) where T
    A,x = M.A,M.B
    xf = paddeddata(x)
    n = max(0,min(length(xf) + bandwidth(A,1),length(M)))
    Vcat(Vector{T}(undef, n), Zeros{T}(size(A,1)-n))
end

function similar(M::MulAdd{<:BandedLayouts,<:AbstractPaddedLayout}, ::Type{T}, axes::Tuple{Any,Any}) where T
    A,x = M.A,M.B
    xf = paddeddata(x)
    m = max(0,min(size(xf,1) + bandwidth(A,1),size(M,1)))
    n = size(xf,2)
    PaddedArray(Matrix{T}(undef, m, n), size(A,1), size(x,2))
end

function materialize!(M::MatMulVecAdd{<:BandedLayouts,<:AbstractPaddedLayout,<:AbstractPaddedLayout})
    α,A,x,β,y = M.α,M.A,M.B,M.β,M.C
    length(y) == size(A,1) || throw(DimensionMismatch())
    length(x) == size(A,2) || throw(DimensionMismatch())

    x̃ = paddeddata(x)
    resizedata!(y, min(length(M),length(x̃)+bandwidth(A,1)))
    ỹ = paddeddata(y)

    muladd!(α, view(A, axes(ỹ,1), axes(x̃,1)) , x̃, β, ỹ)
    y
end

function materialize!(M::MatMulMatAdd{<:BandedLayouts,<:AbstractPaddedLayout,<:AbstractPaddedLayout})
    α,A,x,β,y = M.α,M.A,M.B,M.β,M.C
    size(y) == (size(A,1),size(x,2)) || throw(DimensionMismatch())
    size(x,1) == size(A,2) || throw(DimensionMismatch())

    x̃ = paddeddata(x)
    resizedata!(y, min(size(M,1),size(x̃,1)+bandwidth(A,1)), min(size(M,2),size(x̃,2)))
    ỹ = paddeddata(y)

    if size(ỹ,1) < min(size(M,1),size(x̃,1)+bandwidth(A,1))
        # its ok if the entries are actually zero
        for j = 1:size(x̃,2), k = max(1,size(ỹ,1)-bandwidth(A,1)+1):size(x̃,1)
            iszero(x̃[k,j]) || throw(ArgumentError("Cannot assign non-zero entry $k,$j to zero"))
        end
    end

    muladd!(α, view(A, axes(ỹ,1), axes(x̃,1)) , x̃, β, view(ỹ,:,axes(x̃,2)))
    _fill_lmul!(β, view(ỹ,:,size(x̃,2)+1:size(ỹ,2)))
    y
end

# (vec .* mat) * B is typically faster as vec .* (mat * b)
_broadcast_banded_padded_mul((A1,A2)::Tuple{<:AbstractVector,<:AbstractMatrix}, B) = A1 .* mul(A2, B)
_broadcast_banded_padded_mul(Aargs, B) = copy(mulreduce(Mul(BroadcastArray(*, Aargs...), B)))


###
# MulMatrix
###

bandwidths(M::MulMatrix) = bandwidths(Applied(M))
isbanded(M::Applied{<:Any,typeof(*)}) = all(isbanded, M.args)
isbanded(M::MulMatrix) = isbanded(Applied(M))

###
# ApplyBanded
###

struct ApplyBandedLayout{F} <: AbstractLazyBandedLayout end
struct BroadcastBandedLayout{F} <: AbstractLazyBandedLayout end


bandedlayout(::ApplyLayout{F}) where F = ApplyBandedLayout{F}()
bandedlayout(::BroadcastLayout{F}) where F = BroadcastBandedLayout{F}()
nonbandedlayout(::ApplyBandedLayout{F}) where F = ApplyLayout{F}()
nonbandedlayout(::BroadcastBandedLayout{F}) where F = BroadcastLayout{F}()

arguments(lay::ApplyBandedLayout, A) = arguments(nonbandedlayout(lay), A)


sublayout(lay::ApplyBandedLayout, ind::Type{<:NTuple{2,AbstractUnitRange}}) = bandedlayout(sublayout(nonbandedlayout(lay), ind))
sublayout(lay::ApplyBandedLayout, ind) = sublayout(nonbandedlayout(lay), ind)


LazyArrays._mul_arguments(lay::ApplyBandedLayout, A) = LazyArrays._mul_arguments(nonbandedlayout(lay), A)
@inline islazy_layout(::ApplyBandedLayout) = Val(true)


applylayout(::Type{typeof(*)}, ::BandedLayouts...) = ApplyBandedLayout{typeof(*)}()

applybroadcaststyle(::Type{<:AbstractMatrix}, ::ApplyBandedLayout) = LazyArrayStyle{2}()

@inline colsupport(::ApplyBandedLayout{typeof(*)}, A, j) = banded_colsupport(A, j)
@inline rowsupport(::ApplyBandedLayout{typeof(*)}, A, j) = banded_rowsupport(A, j)

###
# BroadcastMatrix
###

bandwidths(M::BroadcastMatrix) = bandwidths(broadcasted(M))
isbanded(M::BroadcastMatrix) = all(isfinite, bandwidths(M))

BroadcastLayout(::BroadcastBandedLayout{F}) where F = BroadcastLayout{F}()

broadcastlayout(::Type{F}, ::AbstractBandedLayout) where F = BroadcastBandedLayout{F}()

# functions that satisfy f(0,0) == 0

for op in (:+, :-)
    @eval begin
        broadcastlayout(::Type{typeof($op)}, ::AbstractBandedLayout, ::AbstractPaddedLayout) = BroadcastBandedLayout{typeof($op)}()
        broadcastlayout(::Type{typeof($op)}, ::AbstractPaddedLayout, ::AbstractBandedLayout) = BroadcastBandedLayout{typeof($op)}()
    end
end

for op in (:*, :/, :\, :+, :-)
    @eval broadcastlayout(::Type{typeof($op)}, ::AbstractBandedLayout, ::AbstractBandedLayout) = BroadcastBandedLayout{typeof($op)}()
end
for op in (:*, :/)
    @eval broadcastlayout(::Type{typeof($op)}, ::AbstractBandedLayout, ::Any) = BroadcastBandedLayout{typeof($op)}()
end

for op in (:*, :\)
    @eval broadcastlayout(::Type{typeof($op)}, ::Any, ::AbstractBandedLayout) = BroadcastBandedLayout{typeof($op)}()
end

_broadcastarray2broadcasted(::BroadcastBandedLayout{F}, A) where F = _broadcastarray2broadcasted(BroadcastLayout{F}(), A)
_broadcastarray2broadcasted(::BroadcastBandedLayout{F}, A::BroadcastArray) where F = _broadcastarray2broadcasted(BroadcastLayout{F}(), A)


_copyto!(::AbstractBandedLayout, ::BroadcastBandedLayout, dest::AbstractMatrix, bc::AbstractMatrix) =
    copyto!(dest, _broadcastarray2broadcasted(bc))

_copyto!(_, ::BroadcastBandedLayout, dest::AbstractMatrix, bc::AbstractMatrix) =
    copyto!(dest, _broadcastarray2broadcasted(bc))

# _banded_broadcast!(dest::AbstractMatrix, f, (A,B)::Tuple{AbstractMatrix{T},AbstractMatrix{V}}, _, ::Tuple{<:Any,ApplyBandedLayout{typeof(*)}}) where {T,V} =
#     broadcast!(f, dest, BandedMatrix(A), BandedMatrix(B))

broadcasted(::LazyArrayStyle, ::typeof(*), c::Number, A::BandedMatrix) = _BandedMatrix(c .* A.data, A.raxis, A.l, A.u)
broadcasted(::LazyArrayStyle, ::typeof(*), A::BandedMatrix, c::Number) = _BandedMatrix(A.data .* c, A.raxis, A.l, A.u)
broadcasted(::LazyArrayStyle, ::typeof(\), c::Number, A::BandedMatrix) = _BandedMatrix(c .\ A.data, A.raxis, A.l, A.u)
broadcasted(::LazyArrayStyle, ::typeof(/), A::BandedMatrix, c::Number) = _BandedMatrix(A.data ./ c, A.raxis, A.l, A.u)


copy(M::Mul{BroadcastBandedLayout{typeof(*)}, <:AbstractPaddedLayout}) = _broadcast_banded_padded_mul(arguments(BroadcastBandedLayout{typeof(*)}(), M.A), M.B)


###
# copyto!
###

_BandedMatrix(::ApplyBandedLayout{typeof(*)}, V::AbstractMatrix{T}) where T = 
    copyto!(BandedMatrix{T}(undef, axes(V), bandwidths(V)), V)
_BandedMatrix(::BroadcastBandedLayout, V::AbstractMatrix{T}) where T = 
    copyto!(BandedMatrix{T}(undef, axes(V), bandwidths(V)), broadcasted(V))

_broadcast_BandedMatrix(a::AbstractMatrix) = BandedMatrix(a)
_broadcast_BandedMatrix(a) = a

for op in (:+, :-, :*)
    @eval begin
        @inline _BandedMatrix(::BroadcastBandedLayout{typeof($op)}, V::AbstractMatrix)::BandedMatrix = broadcast($op, map(_broadcast_BandedMatrix,arguments(V))...)
        _copyto!(::AbstractBandedLayout, ::BroadcastBandedLayout{typeof($op)}, dest::AbstractMatrix, src::AbstractMatrix) =
            broadcast!($op, dest, map(_broadcast_BandedMatrix, arguments(src))...)
    end
end


_mulbanded_copyto!(dest, a) = copyto!(dest, a)
_mulbanded_copyto!(dest::AbstractArray{T}, a, b) where T = muladd!(one(T), a, b, zero(T), dest)
_mulbanded_copyto!(dest::AbstractArray{T}, a, b, c, d...) where T = _mulbanded_copyto!(dest, mul(a,b), c, d...)

_mulbanded_BandedMatrix(A, _) = A
_mulbanded_BandedMatrix(A, ::NTuple{2,OneTo{Int}}) = BandedMatrix(A)
_mulbanded_BandedMatrix(A) = _mulbanded_BandedMatrix(A, axes(A))

_copyto!(::AbstractBandedLayout, ::ApplyBandedLayout{typeof(*)}, dest::AbstractMatrix, src::AbstractMatrix) =
    _mulbanded_copyto!(dest, map(_mulbanded_BandedMatrix,arguments(src))...)

arguments(::BroadcastBandedLayout{F}, V::SubArray) where F = _broadcast_sub_arguments(V)


call(b::BroadcastBandedLayout, a) = call(BroadcastLayout(b), a)
call(b::BroadcastBandedLayout, a::SubArray) = call(BroadcastLayout(b), a)

sublayout(lay::ApplyBandedLayout{typeof(*)}, ind::Type{<:NTuple{2,AbstractUnitRange}}) = bandedlayout(sublayout(nonbandedlayout(lay), ind))
sublayout(lay::BroadcastBandedLayout, ind::Type{<:NTuple{2,AbstractUnitRange}}) = bandedlayout(sublayout(nonbandedlayout(lay), ind))

transposelayout(b::BroadcastBandedLayout) = b
arguments(b::BroadcastBandedLayout, A::AdjOrTrans) = arguments(BroadcastLayout(b), A)


######
# Concat banded matrix
######


const ZerosLayouts = Union{ZerosLayout,DualLayout{ZerosLayout}}
const ScalarOrZerosLayouts = Union{ScalarLayout,ZerosLayouts}
const ScalarOrBandedLayouts = Union{ScalarOrZerosLayouts,BandedLayouts}

applylayout(::Type{typeof(vcat)}, ::Lay, ::ZerosLayout) where Lay<:BandedLayouts = PaddedColumns{Lay}()
applylayout(::Type{typeof(hcat)}, ::Lay, ::ZerosLayout) where Lay<:BandedLayouts = PaddedRows{Lay}()

applylayout(::Type{typeof(vcat)}, ::BandedLayouts, ::PaddedColumns) = PaddedColumns{ApplyLayout{typeof(vcat)}}()
applylayout(::Type{typeof(hcat)}, ::BandedLayouts, ::PaddedRows) = PaddedRows{ApplyLayout{typeof(hcat)}}()
applylayout(::Type{typeof(hcat)}, ::PaddedColumns, ::BandedLayouts) = ApplyBandedLayout{typeof(hcat)}()

applylayout(::Type{typeof(vcat)}, ::ZerosLayout, ::Lay) where Lay<:BandedLayouts = ApplyBandedLayout{typeof(vcat)}()
applylayout(::Type{typeof(hcat)}, ::ZerosLayout, ::Lay) where Lay<:BandedLayouts = ApplyBandedLayout{typeof(hcat)}()


for op in (:hcat, :vcat)
    @eval sublayout(::ApplyBandedLayout{typeof($op)}, ::Type{<:NTuple{2,AbstractUnitRange}}) = ApplyBandedLayout{typeof($op)}()
end


# some special cases of hvcat we can recognise are banded

const DualPaddedOrZerosRow = DualLayout{<:Union{PaddedRows,ZerosLayout}}
const DualPaddedOrZerosColumn = Union{PaddedColumns,ZerosLayout}

applylayout(::Type{typeof(hvcat)}, _, 
            ::ScalarLayout, ::DualPaddedOrZerosRow,
            ::DualPaddedOrZerosColumn, ::AbstractBandedLayout) = ApplyBandedLayout{typeof(hvcat)}()

applylayout(::Type{typeof(hvcat)}, _, 
            ::ScalarLayout, ::ScalarLayout, ::DualPaddedOrZerosRow,
            ::ScalarLayout, ::ScalarLayout, ::DualPaddedOrZerosRow,
            ::DualPaddedOrZerosColumn, ::DualPaddedOrZerosColumn, ::AbstractBandedLayout) = ApplyBandedLayout{typeof(hvcat)}()


# cumsum for tuples
_cumsum(a) = a
_cumsum(a, b...) = tuple(a, (a .+ _cumsum(b...))...)

_bandwidth(a::Number, n) = iszero(a) ? bandwidth(Zeros{typeof(a)}(1,1),n) : 0
_bandwidth(a, n) = bandwidth(a, n)

_bandwidths(a::Number) = iszero(a) ? bandwidths(Zeros{typeof(a)}(1,1)) : (0,0)
_bandwidths(a) = bandwidths(a)

function bandwidths(M::Vcat{<:Any,2})
    cs = tuple(0, _cumsum(size.(M.args[1:end-1],1)...)...) # cumsum of sizes
    (maximum(cs .+ _bandwidth.(M.args,1)), maximum(_bandwidth.(M.args,2) .- cs))
end
isbanded(M::Vcat) = all(isbanded, M.args)

function bandwidths(M::Hcat)
    cs = tuple(0, _cumsum(size.(M.args[1:end-1],2)...)...) # cumsum of sizes
    (maximum(_bandwidth.(M.args,1) .- cs), maximum(_bandwidth.(M.args,2) .+ cs))
end
isbanded(M::Hcat) = all(isbanded, M.args)

function bandwidths(M::ApplyMatrix{<:Any,typeof(hvcat),<:Tuple{Int,Vararg{Any}}})
    N = first(M.args)
    args = tail(M.args)
    @assert length(args) == N^2
    rs = tuple(0, _cumsum(size.(args[1:N:end-2N+1],1)...)...) # cumsum of sizes
    cs = tuple(0, _cumsum(size.(args[1:N-1],2)...)...) # cumsum of sizes

    l,u = _bandwidth(args[1],1)::Int,_bandwidth(args[1],2)::Int
    for K = 1:N, J = 1:N
        if !(K == J == 1)
            λ,μ = _bandwidth(args[J+N*(K-1)],1),_bandwidth(args[J+N*(K-1)],2)
            if λ ≥ -μ # don't do anything if bandwidths are empty
                l = max(l,λ + rs[K] - cs[J])::Int
                u = max(u,μ + cs[K] - rs[J])::Int
            end
        end
    end
    l,u
end

# just support padded for now
bandwidths(::AbstractPaddedLayout, A) = _bandwidths(paddeddata(A))
isbanded(::AbstractPaddedLayout, A) = true # always treat as banded



const HcatBandedMatrix{T,N} = Hcat{T,NTuple{N,BandedMatrix{T,Matrix{T},OneTo{Int}}}}
const VcatBandedMatrix{T,N} = Vcat{T,2,NTuple{N,BandedMatrix{T,Matrix{T},OneTo{Int}}}}

BroadcastStyle(::Type{HcatBandedMatrix{T,N}}) where {T,N} = BandedStyle()
BroadcastStyle(::Type{VcatBandedMatrix{T,N}}) where {T,N} = BandedStyle()

Base.typed_hcat(::Type{T}, A::BandedMatrix, B::BandedMatrix...) where T = BandedMatrix{T}(Hcat{T}(A, B...))
Base.typed_hcat(::Type{T}, A::BandedMatrix, B::AbstractVecOrMat...) where T = Matrix{T}(Hcat{T}(A, B...))

Base.typed_vcat(::Type{T}, A::BandedMatrix...) where T = BandedMatrix{T}(Vcat{T}(A...))
Base.typed_vcat(::Type{T}, A::BandedMatrix, B::AbstractVecOrMat...) where T = Matrix{T}(Vcat{T}(A, B...))


# layout_broadcasted(lay, ::ApplyBandedLayout{typeof(vcat)}, op, A::AbstractVector, B::AbstractVector) = layout_broadcasted(lay, ApplyLayout{typeof(vcat)}(), op,A, B)
# layout_broadcasted(::ApplyBandedLayout{typeof(vcat)}, lay, op, A::AbstractVector, B::AbstractVector) = layout_broadcasted(ApplyLayout{typeof(vcat)}(), lay, op,A, B)

LazyArrays._vcat_sub_arguments(::ApplyBandedLayout{typeof(vcat)}, A, V) = LazyArrays._vcat_sub_arguments(ApplyLayout{typeof(vcat)}(), A, V)
LazyArrays._vcat_sub_arguments(::ApplyBandedLayout{typeof(hcat)}, A, V) = LazyArrays._vcat_sub_arguments(ApplyLayout{typeof(hcat)}(), A, V)

#######
# CachedArray
#######

CachedArray(::Type{BandedMatrix}, matrix::AbstractMatrix{T}) where T = CachedArray(BandedMatrix{T}(undef, (0,0), bandwidths(matrix)), matrix)

cachedlayout(::BandedColumns{DenseColumnMajor}, ::AbstractBandedLayout) = BandedColumns{DenseColumnMajor}()
bandwidths(B::CachedMatrix) = bandwidths(B.data)
isbanded(B::CachedMatrix) = isbanded(B.data)

function bandeddata(A::CachedMatrix)
    resizedata!(A, size(A)...)
    bandeddata(A.data)
end

function bandeddata(B::SubArray{<:Any,2,<:CachedMatrix})
    A = parent(B)
    kr,jr = parentindices(B)
    resizedata!(A, maximum(kr), maximum(jr))
    bandeddata(view(A.data,kr,jr))
end


function resizedata!(::BandedColumns{DenseColumnMajor}, _, B::AbstractMatrix{T}, n::Integer, m::Integer) where T<:Number
    (n ≤ 0 || m ≤ 0) && return B
    @boundscheck checkbounds(Bool, B, n, m) || throw(ArgumentError("Cannot resize to ($n,$m) which is beyond size $(size(B))"))

    # increase size of array if necessary
    olddata = B.data
    ν,μ = B.datasize
    n,m = max(ν,n), max(μ,m)

    if (ν,μ) ≠ (n,m)
        l,u = bandwidths(B.array)
        λ,ω = bandwidths(B.data)
        if n ≥ size(B.data,1) || m ≥ size(B.data,2)
            M = 2*max(m,n+u)
            B.data = resize(olddata, min(M+λ,size(B,1)), min(M,size(B,2)))
        end
        if ν > 0 # upper-right
            kr = max(1,μ+1-ω):ν
            jr = μ+1:min(m,ν+ω)
            if !isempty(kr) && !isempty(jr)
                view(B.data, kr, jr) .= B.array[kr, jr]
            end
        end
        view(B.data, ν+1:n, μ+1:m) .= B.array[ν+1:n, μ+1:m]
        if μ > 0
            kr = ν+1:min(n,μ+λ)
            jr = max(1,ν+1-λ):μ
            if !isempty(kr) && !isempty(jr)
                view(B.data, kr, jr) .= B.array[kr, jr]
            end
        end
        B.datasize = (n,m)
    end

    B
end

###
# Concat and rot ArrayLayouts
###

applylayout(::Type{typeof(rot180)}, ::BandedColumns{LAY}) where LAY =
    BandedColumns{typeof(sublayout(LAY(), NTuple{2,StepRange{Int,Int}}))}()

applylayout(::Type{typeof(rot180)}, ::AbstractBandedLayout) =
    ApplyBandedLayout{typeof(rot180)}()

call(::ApplyBandedLayout{typeof(*)}, A::ApplyMatrix{<:Any,typeof(rot180)}) = *
applylayout(::Type{typeof(rot180)}, ::ApplyBandedLayout{typeof(*)}) = ApplyBandedLayout{typeof(*)}()
arguments(::ApplyBandedLayout{typeof(*)}, A::ApplyMatrix{<:Any,typeof(rot180)}) = ApplyMatrix.(rot180, arguments(A.args...))


bandwidths(R::ApplyMatrix{<:Any,typeof(rot180)}) = bandwidths(Applied(R))
function bandwidths(R::Applied{<:Any,typeof(rot180)})
    m,n = size(R)
    sh = m-n
    l,u = bandwidths(arguments(R)[1])
    u+sh,l-sh
end

bandeddata(R::ApplyMatrix{<:Any,typeof(rot180)}) = @view(bandeddata(arguments(R)[1])[end:-1:1,end:-1:1])




# leave lazy banded matrices lazy when multiplying.
# overload copy as overloading `mulreduce` requires `copyto!` overloads
# Should probably be redesigned in a trait-based way, but hard to see how to do this

BandedLazyLayouts = Union{AbstractLazyBandedLayout, BandedColumns{LazyLayout}, BandedRows{LazyLayout},
TriangularLayout{UPLO,UNIT,BandedRows{LazyLayout}} where {UPLO,UNIT},
TriangularLayout{UPLO,UNIT,BandedColumns{LazyLayout}} where {UPLO,UNIT},
SymTridiagonalLayout{LazyLayout}, BidiagonalLayout{LazyLayout}, TridiagonalLayout{LazyLayout},
SymmetricLayout{BandedColumns{LazyLayout}}, HermitianLayout{BandedColumns{LazyLayout}}}

@inline islazy_layout(::BandedLazyLayouts) = Val(true)

copy(M::Mul{<:BandedLazyLayouts, <:BandedLazyLayouts}) = simplify(M)
copy(M::Mul{<:BandedLazyLayouts}) = simplify(M)
copy(M::Mul{<:Any, <:BandedLazyLayouts}) = simplify(M)
copy(M::Mul{<:BandedLazyLayouts, <:AbstractLazyLayout}) = simplify(M)
copy(M::Mul{<:AbstractLazyLayout, <:BandedLazyLayouts}) = simplify(M)
copy(M::Mul{BroadcastLayout{typeof(*)}, <:BandedLazyLayouts}) = _broadcast_mul_mul(arguments(BroadcastLayout{typeof(*)}(), M.A), M.B)
copy(M::Mul{<:BandedLazyLayouts, <:DiagonalLayout}) = simplify(M)
copy(M::Mul{<:DiagonalLayout, <:BandedLazyLayouts}) = simplify(M)


copy(M::Mul{<:Union{ZerosLayout,DualLayout{ZerosLayout}}, <:BandedLazyLayouts}) = copy(mulreduce(M))
copy(M::Mul{<:BandedLazyLayouts, <:Union{ZerosLayout,DualLayout{ZerosLayout}}}) = copy(mulreduce(M))

simplifiable(::Mul{<:BandedLazyLayouts, <:DiagonalLayout{<:OnesLayout}}) = Val(true)
simplifiable(::Mul{<:DiagonalLayout{<:OnesLayout}, <:BandedLazyLayouts}) = Val(true)
copy(M::Mul{<:BandedLazyLayouts, <:DiagonalLayout{<:OnesLayout}}) = _copy_oftype(M.A, eltype(M))
copy(M::Mul{<:DiagonalLayout{<:OnesLayout}, <:BandedLazyLayouts}) = _copy_oftype(M.B, eltype(M))

copy(M::Mul{<:DiagonalLayout{<:AbstractFillLayout}, <:BandedLazyLayouts}) = copy(mulreduce(M))
copy(M::Mul{<:BandedLazyLayouts, <:DiagonalLayout{<:AbstractFillLayout}}) = copy(mulreduce(M))

BandedAndBroadcastLayouts{F} = Union{BroadcastLayout{F},BroadcastBandedLayout{F}}

copy(M::Mul{ApplyBandedLayout{typeof(*)},ApplyBandedLayout{typeof(*)}}) = simplify(M)
copy(M::Mul{ApplyBandedLayout{typeof(*)},<:BandedLazyLayouts}) = simplify(M)
copy(M::Mul{<:BandedLazyLayouts,ApplyBandedLayout{typeof(*)}}) = simplify(M)
copy(M::Mul{ApplyBandedLayout{typeof(*)},<:BandedAndBroadcastLayouts}) = simplify(M)
copy(M::Mul{<:BandedAndBroadcastLayouts,ApplyBandedLayout{typeof(*)}}) = simplify(M)
copy(M::Mul{BroadcastLayout{typeof(*)},ApplyBandedLayout{typeof(*)}}) = simplify(M)
copy(M::Mul{ApplyLayout{typeof(*)},<:BandedLazyLayouts}) = simplify(M)
copy(M::Mul{<:BandedLazyLayouts,ApplyLayout{typeof(*)}}) = simplify(M)
copy(M::Mul{ApplyLayout{typeof(*)},<:BandedAndBroadcastLayouts}) = simplify(M)
copy(M::Mul{<:BandedAndBroadcastLayouts,ApplyLayout{typeof(*)}}) = simplify(M)

copy(M::Mul{<:AbstractInvLayout, ApplyBandedLayout{typeof(*)}}) = simplify(M)
simplifiable(::Mul{<:AbstractInvLayout, <:BandedLazyLayouts}) = Val(false)
copy(M::Mul{<:AbstractInvLayout, <:BandedLazyLayouts}) = simplify(M)


copy(L::Ldiv{<:BandedLazyLayouts, <:BandedLazyLayouts}) = lazymaterialize(\, L.A, L.B)

# TODO: this is type piracy
function colsupport(lay::ApplyLayout{typeof(\)}, L, j)
    A,B = arguments(lay, L)
    l,u = bandwidths(A)
    cs = colsupport(B,j)
    m = size(L,1)
    l == u == 0 && return cs
    l == 0 && return 1:last(cs)
    u == 0 && return first(cs):m
    1:m
end

function rowsupport(lay::ApplyLayout{typeof(\)}, L, k)
    A,B = arguments(lay, L)
    l,u = bandwidths(A)
    cs = rowsupport(B,k)
    m = size(L,1)
    l == u == 0 && return cs
    l == 0 && return first(cs):m
    u == 0 && return 1:last(cs)
    1:m
end

copy(M::Mul{ApplyLayout{typeof(\)}, <:BroadcastBandedLayout}) = lazymaterialize(*, M.A, M.B)
copy(M::Mul{BroadcastLayout{typeof(*)}, <:BroadcastBandedLayout}) = lazymaterialize(*, M.A, M.B)

## padded copy
mulreduce(M::Mul{<:BroadcastBandedLayout, <:Union{AbstractPaddedLayout,AbstractStridedLayout}}) = MulAdd(M)
mulreduce(M::Mul{ApplyBandedLayout{F}, D}) where {F,D<:Union{AbstractPaddedLayout,AbstractStridedLayout}} = Mul{ApplyLayout{F},D}(M.A, M.B)
# need to overload copy due to above
copy(M::Mul{<:BroadcastBandedLayout, <:Union{AbstractPaddedLayout,AbstractStridedLayout}}) = copy(mulreduce(M))
copy(M::Mul{<:AbstractInvLayout{<:BandedLazyLayouts}, <:Union{AbstractPaddedLayout,AbstractStridedLayout}}) = ArrayLayouts.ldiv(pinv(M.A), M.B)
copy(M::Mul{<:BandedLazyLayouts, <:Union{AbstractPaddedLayout,AbstractStridedLayout}}) = copy(mulreduce(M))
copy(M::Mul{<:Union{AbstractPaddedLayout,AbstractStridedLayout}, <:BandedLazyLayouts}) = copy(mulreduce(M))

simplifiable(M::Mul{<:AbstractInvLayout{<:BandedLazyLayouts}, <:Union{AbstractPaddedLayout,AbstractStridedLayout}}) = Val(true)
simplifiable(M::Mul{<:Union{AbstractPaddedLayout,AbstractStridedLayout}, <:BandedLazyLayouts}) = Val(true)
simplifiable(M::Mul{<:BandedLazyLayouts, <:Union{AbstractPaddedLayout,AbstractStridedLayout}}) = Val(true)
simplifiable(::Mul{<:BroadcastBandedLayout, <:Union{AbstractPaddedLayout,AbstractStridedLayout}}) = Val(true)

copy(L::Ldiv{ApplyBandedLayout{typeof(*)}, Lay}) where Lay = copy(Ldiv{ApplyLayout{typeof(*)},Lay}(L.A, L.B))
copy(L::Ldiv{ApplyBandedLayout{typeof(*)}, Lay}) where Lay<:BroadcastBandedLayout = copy(Ldiv{ApplyLayout{typeof(*)},Lay}(L.A, L.B))
_inv(::BandedLazyLayouts, _, A) = ApplyArray(inv, A)

####
# Band getindex
####

function getindex(bc::BroadcastArray{<:Any,2,<:Any,<:NTuple{2,AbstractMatrix}}, b::Band)
    A,B = bc.args
    bc.f.(A[b],B[b])
end
function getindex(bc::BroadcastArray{<:Any,2,<:Any,<:Tuple{Number,AbstractMatrix}}, b::Band)
    a,B = bc.args
    bc.f.(a,B[b])
end
function getindex(bc::BroadcastArray{<:Any,2,<:Any,<:Tuple{AbstractMatrix,Number}}, b::Band)
    A,c = bc.args
    bc.f.(A[b],c)
end



end
