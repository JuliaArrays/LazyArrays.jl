module LazyArrays

# Use README as the docstring of the module:
@doc read(joinpath(dirname(@__DIR__), "README.md"), String) LazyArrays

using Base, Base.Broadcast, LinearAlgebra, FillArrays, StaticArrays, ArrayLayouts
import LinearAlgebra.BLAS

import Base: AbstractArray, AbstractMatrix, AbstractVector,
        ReinterpretArray, ReshapedArray, AbstractCartesianIndex, Slice,
             RangeIndex, BroadcastStyle, copyto!, length, broadcastable, axes,
             getindex, eltype, tail, IndexStyle, IndexLinear, getproperty,
             *, +, -, /, \, ==, isinf, isfinite, sign, angle, show, isless,
         fld, cld, div, min, max, minimum, maximum, mod,
         <, ≤, >, ≥, promote_rule, convert, copy,
         size, step, isempty, length, first, last, ndims,
         getindex, setindex!, intersect, @_inline_meta, inv,
         sort, sort!, issorted, sortperm, diff, cumsum, sum, in, broadcast,
         eltype, parent, real, imag,
         conj, transpose, adjoint, permutedims, vec,
         exp, log, sqrt, cos, sin, tan, csc, sec, cot,
                   cosh, sinh, tanh, csch, sech, coth,
                   acos, asin, atan, acsc, asec, acot,
                   acosh, asinh, atanh, acsch, asech, acoth, (:),
         AbstractMatrix, AbstractArray, checkindex, unsafe_length, OneTo, one, zero,
        to_shape, _sub2ind, print_matrix, print_matrix_row, print_matrix_vdots,
      checkindex, Slice, @propagate_inbounds, @_propagate_inbounds_meta,
      _in_range, _range, _rangestyle, Ordered,
      ArithmeticWraps, floatrange, reverse, unitrange_last,
      AbstractArray, AbstractVector, axes, (:), _sub2ind_recurse, broadcast, promote_eltypeof,
      similar, @_gc_preserve_end, @_gc_preserve_begin,
      @nexprs, @ncall, @ntuple, tuple_type_tail,
      all, any, isbitsunion, issubset, replace_in_print_matrix, replace_with_centered_mark

import Base.Broadcast: BroadcastStyle, AbstractArrayStyle, Broadcasted, broadcasted,
                        combine_eltypes, DefaultArrayStyle, instantiate, materialize,
                        materialize!, eltypes

import LinearAlgebra: AbstractTriangular, AbstractQ, checksquare, pinv, fill!, tilebufsize, Abuf, Bbuf, Cbuf, dot, factorize, qr, lu, cholesky,
                        norm2, norm1, normInf, normp, normMinusInf, det, tr, AdjOrTrans

import LinearAlgebra.BLAS: BlasFloat, BlasReal, BlasComplex

import FillArrays: AbstractFill, getindex_value

import StaticArrays: StaticArrayStyle

import ArrayLayouts: MatMulVecAdd, MatMulMatAdd, MulAdd, Lmul, Rmul, Ldiv, 
                        transposelayout, conjlayout, sublayout, triangularlayout, triangulardata,
                        reshapedlayout, diagonallayout, adjointlayout, sub_materialize,
                        check_mul_axes, _mul_eltype, check_ldiv_axes, ldivaxes, colsupport, rowsupport,
                        _fill_lmul!, @lazylmul, scalarone, scalarzero, fillzeros, zero!, lazy_getindex

if VERSION < v"1.2-"
    import Base: has_offset_axes
    require_one_based_indexing(A...) = !has_offset_axes(A...) || throw(ArgumentError("offset arrays are not supported but got an array with index other than 1"))
else
    import Base: require_one_based_indexing
end

export Mul, Applied, MulArray, MulVector, MulMatrix, InvMatrix, PInvMatrix,
        Hcat, Vcat, Kron, BroadcastArray, BroadcastMatrix, BroadcastVector, cache, Ldiv, Inv, PInv, Diff, Cumsum,
        applied, materialize, materialize!, ApplyArray, ApplyMatrix, ApplyVector, apply, ⋆, @~, LazyArray


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

end # module
