module LazyArrays

# Use README as the docstring of the module:
@doc read(joinpath(dirname(@__DIR__), "README.md"), String) LazyArrays

using Base, Base.Broadcast, LinearAlgebra, FillArrays, StaticArrays, ArrayLayouts, MatrixFactorizations, SparseArrays
import LinearAlgebra.BLAS

import Base: AbstractArray, AbstractMatrix, AbstractVector,
        ReinterpretArray, ReshapedArray, AbstractCartesianIndex, Slice,
             RangeIndex, BroadcastStyle, copyto!, length, broadcastable, axes,
             getindex, eltype, tail, IndexStyle, IndexLinear, getproperty,
             *, +, -, /, \, ==, isinf, isfinite, sign, angle, show, isless,
         fld, cld, div, min, max, minimum, maximum, mod,
         <, ≤, >, ≥, promote_rule, convert, copy,
         size, step, isempty, length, first, last, ndims,
         getindex, setindex!, setindex, intersect, @_inline_meta, inv,
         sort, sort!, issorted, sortperm, diff, accumulate, cumsum, sum, in, broadcast,
         eltype, parent, real, imag,
         conj, transpose, adjoint, permutedims, vec,
         exp, log, sqrt, cos, sin, tan, csc, sec, cot,
                   cosh, sinh, tanh, csch, sech, coth,
                   acos, asin, atan, acsc, asec, acot,
                   acosh, asinh, atanh, acsch, asech, acoth, (:),
         AbstractMatrix, AbstractArray, checkindex, unsafe_length, OneTo, one, zero,
        to_shape, _sub2ind, print_matrix, print_matrix_row, print_matrix_vdots,
      checkindex, Slice, @propagate_inbounds, @_propagate_inbounds_meta,
      _in_range, _range, Ordered,
      ArithmeticWraps, floatrange, reverse, unitrange_last,
      AbstractArray, AbstractVector, axes, (:), _sub2ind_recurse, broadcast, promote_eltypeof,
      similar, @_gc_preserve_end, @_gc_preserve_begin,
      @nexprs, @ncall, @ntuple, tuple_type_tail,
      all, any, isbitsunion, issubset, replace_with_centered_mark, replace_in_print_matrix,
      unsafe_convert, strides, union, map, searchsortedfirst, searchsortedlast, searchsorted

import Base.Broadcast: BroadcastStyle, AbstractArrayStyle, Broadcasted, broadcasted,
                        combine_eltypes, DefaultArrayStyle, instantiate, materialize,
                        materialize!, eltypes

import LinearAlgebra: AbstractTriangular, AbstractQ, checksquare, pinv, fill!, tilebufsize, dot, factorize, qr, lu, cholesky,
                        norm2, norm1, normInf, normp, normMinusInf, diag, det, logabsdet, tr, AdjOrTrans, triu, tril,
                        lmul!, rmul!

import LinearAlgebra.BLAS: BlasFloat, BlasReal, BlasComplex

import FillArrays: AbstractFill, getindex_value

import StaticArrays: StaticArrayStyle

import ArrayLayouts: MatMulVecAdd, MatMulMatAdd, MulAdd, Lmul, Rmul, Ldiv, Dot, Mul, _inv,
                        transposelayout, conjlayout, sublayout, triangularlayout, triangulardata,
                        reshapedlayout, diagonallayout, tridiagonallayout, symtridiagonallayout, bidiagonallayout, symmetriclayout, hermitianlayout,
                        adjointlayout, sub_materialize, mulreduce,
                        check_mul_axes, _mul_eltype, check_ldiv_axes, ldivaxes, colsupport, rowsupport,
                        _fill_lmul!, scalarone, scalarzero, fillzeros, zero!, layout_getindex, _copyto!,
                        AbstractQLayout, StridedLayout, layout_replace_in_print_matrix

import Base: require_one_based_indexing, oneto

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

# support x^2
Base.to_power_type(x::LazyArray) = x

# Special broadcasting for BlockArrays.jl
map(::typeof(length), A::BroadcastArray{OneTo{Int},1,Type{OneTo}}) = A.args[1]
map(::typeof(length), A::BroadcastArray{<:Fill,1,Type{Fill},<:Tuple{Any,AbstractVector}}) = A.args[2]
map(::typeof(length), A::BroadcastArray{<:Fill,1,Type{Fill},<:Tuple{AbstractVector,Number}}) = Fill(A.args[2],length(A.args[1]))
map(::typeof(length), A::BroadcastArray{<:Zeros,1,Type{Zeros}}) = A.args[1]
map(::typeof(length), A::BroadcastArray{<:Vcat,1,Type{Vcat}}) = broadcast(+,map.(length,A.args)...)
broadcasted(::LazyArrayStyle{1}, ::typeof(length), A::BroadcastArray{OneTo{Int},1,Type{OneTo}}) = A.args[1]
broadcasted(::LazyArrayStyle{1}, ::typeof(length), A::BroadcastArray{<:Fill,1,Type{Fill},<:NTuple{2,Any}}) = A.args[2]

end # module
