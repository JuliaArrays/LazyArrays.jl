module LazyArrays

# Use README as the docstring of the module:
@doc read(joinpath(dirname(@__DIR__), "README.md"), String) LazyArrays

using Base.Broadcast, LinearAlgebra, FillArrays, ArrayLayouts, SparseArrays
import LinearAlgebra.BLAS

import Base: *, +, -, /, <, ==, >, \, ≤, ≥, (:), @_gc_preserve_begin, @_gc_preserve_end, @propagate_inbounds,
             AbstractArray, AbstractMatrix, AbstractVector, BroadcastStyle, IndexLinear, IndexStyle, OneTo, Slice,
             accumulate, acos, acosh, acot, acoth, acsc, acsch, adjoint, all, any, asec, asech, asin, asinh, atan,
             atanh, axes, broadcast, broadcastable, conj, convert, copy, copyto!, cos, cosh, cot, coth, csc, csch,
             cumsum, diff, div, eltype, exp, first, getindex, in, intersect, inv, isbitsunion, isempty, isinf, issubset,
             last, length, log, map, max, maximum, min, minimum, mod, ndims, one, parent, permutedims, print_matrix,
             real, replace_in_print_matrix, replace_with_centered_mark, reverse, searchsorted, searchsortedfirst,
             searchsortedlast, sec, sech, setindex, setindex!, show, similar, sin, sinh, size, sort, sqrt, strides, sum,
             tail, tan, tanh, transpose, tuple_type_tail, union, unsafe_convert, vec, zero, fill!, require_one_based_indexing,
             oneto, add_sum, promote_op

import Base.Broadcast: AbstractArrayStyle, BroadcastStyle, Broadcasted, DefaultArrayStyle, broadcasted, combine_eltypes,
                       instantiate

import LinearAlgebra: AbstractQ, AdjOrTrans, StructuredMatrixStyle, checksquare, det, diag, dot, lmul!, logabsdet,
                      norm1, norm2, normInf, normp, pinv, rmul!, tr, tril, triu

if VERSION ≥ v"1.11.0-DEV.21"
    using LinearAlgebra: UpperOrLowerTriangular
else
    const UpperOrLowerTriangular{T,S} = Union{LinearAlgebra.UpperTriangular{T,S},
                                              LinearAlgebra.UnitUpperTriangular{T,S},
                                              LinearAlgebra.LowerTriangular{T,S},
                                              LinearAlgebra.UnitLowerTriangular{T,S}}
end

import ArrayLayouts: AbstractQLayout, Dot, Dotu, Ldiv, Lmul, MatMulMatAdd, MatMulVecAdd, Mul, MulAdd, Rmul,
                     StridedLayout, copyto!_layout, _fill_lmul!, inv_layout, _mul_eltype, adjointlayout, bidiagonallayout,
                     check_ldiv_axes, check_mul_axes, colsupport, conjlayout, diagonallayout, dotu, fillzeros,
                     hermitianlayout, layout_getindex, layout_replace_in_print_matrix, ldivaxes, materialize,
                     materialize!, mulreduce, reshapedlayout, rowsupport, scalarone, scalarzero, sub_materialize,
                     sublayout, symmetriclayout, symtridiagonallayout, transposelayout, triangulardata,
                     triangularlayout, tridiagonallayout, zero!, transtype, OnesLayout,
                     diagonaldata, subdiagonaldata, supdiagonaldata, MemoryLayout

import FillArrays: AbstractFill, getindex_value

export Mul, Applied, MulArray, MulVector, MulMatrix, InvMatrix, PInvMatrix,
        Hcat, Vcat, Kron, BroadcastArray, BroadcastMatrix, BroadcastVector, cache, Ldiv, Inv, PInv, Diff, Cumsum, Accumulate,
        applied, materialize, materialize!, ApplyArray, ApplyMatrix, ApplyVector, apply, @~, LazyArray,
        PaddedArray, PaddedVector, PaddedMatrix


include("lazyapplying.jl")
include("lazybroadcasting.jl")
include("linalg/linalg.jl")
include("cache.jl")
include("lazyconcat.jl")
include("lazysetoperations.jl")
include("lazyoperations.jl")
include("lazymacro.jl")

if isdefined(Base, :to_power_type)
    # Gone in Julia 1.12-rc1
    # support x^2
    Base.to_power_type(x::LazyArray) = x
end

# Special broadcasting for BlockArrays.jl
map(::typeof(length), A::BroadcastVector{OneTo{Int},Type{OneTo}}) = A.args[1]
map(::typeof(length), A::BroadcastVector{<:Fill,Type{Fill},<:Tuple{Any,AbstractVector}}) = A.args[2]
map(::typeof(length), A::BroadcastVector{<:Fill,Type{Fill},<:Tuple{AbstractVector,Number}}) = Fill(A.args[2],length(A.args[1]))
map(::typeof(length), A::BroadcastVector{<:OneElement,Type{OneElement},<:Tuple{Any,AbstractVector}}) = A.args[2]
map(::typeof(length), A::BroadcastVector{<:OneElement,Type{OneElement},<:Tuple{AbstractVector,Number}}) = Fill(A.args[2],length(A.args[1]))
map(::typeof(length), A::BroadcastVector{<:Zeros,Type{Zeros}}) = A.args[1]
map(::typeof(length), A::BroadcastVector{<:Vcat,Type{Vcat}}) = broadcast(+,map.(length,A.args)...)
broadcasted(::LazyArrayStyle{1}, ::typeof(length), A::BroadcastVector{OneTo{Int},Type{OneTo}}) = A.args[1]
broadcasted(::LazyArrayStyle{1}, ::typeof(length), A::BroadcastVector{<:Fill,Type{Fill},<:NTuple{2,Any}}) = A.args[2]

# types for use by extensions
function _mulbanded_copyto! end

abstract type AbstractLazyBandedLayout <: AbstractBandedLayout end
struct LazyBandedLayout <: AbstractLazyBandedLayout end
struct ApplyBandedLayout{F} <: AbstractLazyBandedLayout end
struct BroadcastBandedLayout{F} <: AbstractLazyBandedLayout end


end # module
