abstract type MemoryLayout end
struct UnknownLayout <: MemoryLayout end
abstract type AbstractStridedLayout <: MemoryLayout end
abstract type AbstractIncreasingStrides <: AbstractStridedLayout end
abstract type AbstractColumnMajor <: AbstractIncreasingStrides end
struct DenseColumnMajor <: AbstractColumnMajor end
struct ColumnMajor <: AbstractColumnMajor end
struct IncreasingStrides <: AbstractIncreasingStrides end
abstract type AbstractDecreasingStrides <: AbstractStridedLayout end
abstract type AbstractRowMajor <: AbstractDecreasingStrides end
struct DenseRowMajor <: AbstractRowMajor end
struct RowMajor <: AbstractRowMajor end
struct DecreasingStrides <: AbstractIncreasingStrides end
struct StridedLayout <: AbstractStridedLayout end
struct ScalarLayout <: MemoryLayout end

"""
    UnknownLayout()

is returned by `MemoryLayout(A)` if it is unknown how the entries of an array `A`
are stored in memory.
"""
UnknownLayout

"""
    AbstractStridedLayout

is an abstract type whose subtypes are returned by `MemoryLayout(A)`
if an array `A` has storage laid out at regular offsets in memory,
and which can therefore be passed to external C and Fortran functions expecting
this memory layout.

Julia's internal linear algebra machinery will automatically (and invisibly)
dispatch to BLAS and LAPACK routines if the memory layout is BLAS compatible and
the element type is a `Float32`, `Float64`, `ComplexF32`, or `ComplexF64`.
In this case, one must implement the strided array interface, which requires
overrides of `strides(A::MyMatrix)` and `unknown_convert(::Type{Ptr{T}}, A::MyMatrix)`.
"""
AbstractStridedLayout

"""
    DenseColumnMajor()

is returned by `MemoryLayout(A)` if an array `A` has storage in memory
equivalent to an `Array`, so that `stride(A,1) == 1` and
`stride(A,i) ≡ size(A,i-1) * stride(A,i-1)` for `2 ≤ i ≤ ndims(A)`. In particular,
if `A` is a matrix then `strides(A) == `(1, size(A,1))`.

Arrays with `DenseColumnMajor` memory layout must conform to the `DenseArray` interface.
"""
DenseColumnMajor

"""
    ColumnMajor()

is returned by `MemoryLayout(A)` if an array `A` has storage in memory
as a column major array, so that `stride(A,1) == 1` and
`stride(A,i) ≥ size(A,i-1) * stride(A,i-1)` for `2 ≤ i ≤ ndims(A)`.

Arrays with `ColumnMajor` memory layout must conform to the `DenseArray` interface.
"""
ColumnMajor

"""
    IncreasingStrides()

is returned by `MemoryLayout(A)` if an array `A` has storage in memory
as a strided array with  increasing strides, so that `stride(A,1) ≥ 1` and
`stride(A,i) ≥ size(A,i-1) * stride(A,i-1)` for `2 ≤ i ≤ ndims(A)`.
"""
IncreasingStrides

"""
    DenseRowMajor()

is returned by `MemoryLayout(A)` if an array `A` has storage in memory
as a row major array with dense entries, so that `stride(A,ndims(A)) == 1` and
`stride(A,i) ≡ size(A,i+1) * stride(A,i+1)` for `1 ≤ i ≤ ndims(A)-1`. In particular,
if `A` is a matrix then `strides(A) == `(size(A,2), 1)`.
"""
DenseRowMajor

"""
    RowMajor()

is returned by `MemoryLayout(A)` if an array `A` has storage in memory
as a row major array, so that `stride(A,ndims(A)) == 1` and
stride(A,i) ≥ size(A,i+1) * stride(A,i+1)` for `1 ≤ i ≤ ndims(A)-1`.

If `A` is a matrix  with `RowMajor` memory layout, then
`transpose(A)` should return a matrix whose layout is `ColumnMajor`.
"""
RowMajor

"""
    DecreasingStrides()

is returned by `MemoryLayout(A)` if an array `A` has storage in memory
as a strided array with decreasing strides, so that `stride(A,ndims(A)) ≥ 1` and
stride(A,i) ≥ size(A,i+1) * stride(A,i+1)` for `1 ≤ i ≤ ndims(A)-1`.
"""
DecreasingStrides

"""
    StridedLayout()

is returned by `MemoryLayout(A)` if an array `A` has storage laid out at regular
offsets in memory. `Array`s with `StridedLayout` must conform to the `DenseArray` interface.
"""
StridedLayout

"""
    ScalarLayout()

is returned by `MemoryLayout(A)` if A is a scalar, which does not live in memory
"""
ScalarLayout

"""
    MemoryLayout(A)

specifies the layout in memory for an array `A`. When
you define a new `AbstractArray` type, you can choose to override
`MemoryLayout` to indicate how an array is stored in memory.
For example, if your matrix is column major with `stride(A,2) == size(A,1)`,
then override as follows:

    MemoryLayout(::MyMatrix) = DenseColumnMajor()

The default is `UnknownLayout()` to indicate that the layout
in memory is unknown.

Julia's internal linear algebra machinery will automatically (and invisibly)
dispatch to BLAS and LAPACK routines if the memory layout is compatible.
"""
@inline MemoryLayout(_) = UnknownLayout()

@inline MemoryLayout(::Number) = ScalarLayout()
@inline MemoryLayout(::DenseArray) = DenseColumnMajor()

@inline MemoryLayout(A::ReinterpretArray) = reinterpretedmemorylayout(MemoryLayout(parent(A)))
@inline reinterpretedmemorylayout(::MemoryLayout) = UnknownLayout()
@inline reinterpretedmemorylayout(::DenseColumnMajor) = DenseColumnMajor()

@inline MemoryLayout(A::ReshapedArray) = reshapedmemorylayout(MemoryLayout(parent(A)))
@inline reshapedmemorylayout(::MemoryLayout) = UnknownLayout()
@inline reshapedmemorylayout(::DenseColumnMajor) = DenseColumnMajor()


@inline MemoryLayout(A::SubArray) = subarraylayout(MemoryLayout(parent(A)), parentindices(A))
subarraylayout(_1, _2) = UnknownLayout()
subarraylayout(_1, _2, _3)= UnknownLayout()
subarraylayout(::DenseColumnMajor, ::Tuple{I}) where I<:Union{AbstractUnitRange{Int},Int,AbstractCartesianIndex} =
    DenseColumnMajor()  # A[:] is DenseColumnMajor if A is DenseColumnMajor
subarraylayout(ml::AbstractColumnMajor, inds) = _column_subarraylayout1(ml, inds)
subarraylayout(::AbstractRowMajor, ::Tuple{I}) where I =
    UnknownLayout()  # A[:] does not have any structure if A is AbstractRowMajor
subarraylayout(ml::AbstractRowMajor, inds) = _row_subarraylayout1(ml, reverse(inds))
subarraylayout(ml::AbstractStridedLayout, inds) = _strided_subarraylayout(ml, inds)

_column_subarraylayout1(::DenseColumnMajor, inds::Tuple{I,Vararg{Int}}) where I<:Union{Int,AbstractCartesianIndex} =
    DenseColumnMajor() # view(A,1,1,2) is a scalar, which we include in DenseColumnMajor
_column_subarraylayout1(::DenseColumnMajor, inds::Tuple{I,Vararg{Int}}) where I<:Slice =
    DenseColumnMajor() # view(A,:,1,2) is a DenseColumnMajor vector
_column_subarraylayout1(::DenseColumnMajor, inds::Tuple{I,Vararg{Int}}) where I<:AbstractUnitRange{Int} =
    DenseColumnMajor() # view(A,1:3,1,2) is a DenseColumnMajor vector
_column_subarraylayout1(par, inds::Tuple{I,Vararg{Int}}) where I<:Union{Int,AbstractCartesianIndex} =
    DenseColumnMajor() # view(A,1,1,2) is a scalar, which we include in DenseColumnMajor
_column_subarraylayout1(par, inds::Tuple{I,Vararg{Int}}) where I<:AbstractUnitRange{Int} =
    DenseColumnMajor() # view(A,1:3,1,2) is a DenseColumnMajor vector
_column_subarraylayout1(::DenseColumnMajor, inds::Tuple{I,Vararg{Any}}) where I<:Slice =
    _column_subarraylayout(DenseColumnMajor(), DenseColumnMajor(), tail(inds))
_column_subarraylayout1(par::DenseColumnMajor, inds::Tuple{I,Vararg{Any}}) where I<:AbstractUnitRange{Int} =
    _column_subarraylayout(par, ColumnMajor(), tail(inds))
_column_subarraylayout1(par, inds::Tuple{I,Vararg{Any}}) where I<:AbstractUnitRange{Int} =
    _column_subarraylayout(par, ColumnMajor(), tail(inds))
_column_subarraylayout1(par::DenseColumnMajor, inds::Tuple{I,Vararg{Any}}) where I<:Union{RangeIndex,AbstractCartesianIndex} =
    _column_subarraylayout(par, StridedLayout(), tail(inds))
_column_subarraylayout1(par, inds::Tuple{I,Vararg{Any}}) where I<:Union{RangeIndex,AbstractCartesianIndex} =
    _column_subarraylayout(par, StridedLayout(), tail(inds))
_column_subarraylayout1(par, inds) = UnknownLayout()
_column_subarraylayout(par, ret, ::Tuple{}) = ret
_column_subarraylayout(par, ret, ::Tuple{I}) where I = UnknownLayout()
_column_subarraylayout(::DenseColumnMajor, ::DenseColumnMajor, inds::Tuple{I,Vararg{Int}}) where I<:Union{AbstractUnitRange{Int},Int,AbstractCartesianIndex} =
    DenseColumnMajor() # A[:,1:3,1,2] is DenseColumnMajor if A is DenseColumnMajor
_column_subarraylayout(par::DenseColumnMajor, ::DenseColumnMajor, inds::Tuple{I, Vararg{Int}}) where I<:Slice =
    DenseColumnMajor()
_column_subarraylayout(par::DenseColumnMajor, ::DenseColumnMajor, inds::Tuple{I, Vararg{Any}}) where I<:Slice =
    _column_subarraylayout(par, DenseColumnMajor(), tail(inds))
_column_subarraylayout(par, ::AbstractColumnMajor, inds::Tuple{I, Vararg{Any}}) where I<:Union{AbstractUnitRange{Int},Int,AbstractCartesianIndex} =
    _column_subarraylayout(par, ColumnMajor(), tail(inds))
_column_subarraylayout(par, ::AbstractStridedLayout, inds::Tuple{I, Vararg{Any}}) where I<:Union{RangeIndex,AbstractCartesianIndex} =
    _column_subarraylayout(par, StridedLayout(), tail(inds))

_row_subarraylayout1(par, inds::Tuple{I,Vararg{Int}}) where I<:Union{Int,AbstractCartesianIndex} =
    DenseColumnMajor() # view(A,1,1,2) is a scalar, which we include in DenseColumnMajor
_row_subarraylayout1(::DenseRowMajor, inds::Tuple{I,Vararg{Int}}) where I<:Slice =
    DenseColumnMajor() # view(A,1,2,:) is a DenseColumnMajor vector
_row_subarraylayout1(par, inds::Tuple{I,Vararg{Int}}) where I<:AbstractUnitRange{Int} =
    DenseColumnMajor() # view(A,1,2,1:3) is a DenseColumnMajor vector
_row_subarraylayout1(::DenseRowMajor, inds::Tuple{I,Vararg{Any}}) where I<:Slice =
    _row_subarraylayout(DenseRowMajor(), DenseRowMajor(), tail(inds))
_row_subarraylayout1(par, inds::Tuple{I,Vararg{Any}}) where I<:AbstractUnitRange{Int} =
    _row_subarraylayout(par, RowMajor(), tail(inds))
_row_subarraylayout1(par, inds::Tuple{I,Vararg{Any}}) where I<:Union{RangeIndex,AbstractCartesianIndex} =
    _row_subarraylayout(par, StridedLayout(), tail(inds))
_row_subarraylayout1(par, inds) = UnknownLayout()
_row_subarraylayout(par, ret, ::Tuple{}) = ret
_row_subarraylayout(par, ret, ::Tuple{I}) where I = UnknownLayout()
_row_subarraylayout(::DenseRowMajor, ::DenseRowMajor, inds::Tuple{I,Vararg{Int}}) where I<:Union{AbstractUnitRange{Int},Int,AbstractCartesianIndex} =
    DenseRowMajor() # A[1,2,1:3,:] is DenseRowMajor if A is DenseRowMajor
_row_subarraylayout(par::DenseRowMajor, ::DenseRowMajor, inds::Tuple{I, Vararg{Int}}) where I<:Slice =
    DenseRowMajor()
_row_subarraylayout(par::DenseRowMajor, ::DenseRowMajor, inds::Tuple{I, Vararg{Any}}) where I<:Slice =
    _row_subarraylayout(par, DenseRowMajor(), tail(inds))
_row_subarraylayout(par::AbstractRowMajor, ::AbstractRowMajor, inds::Tuple{I, Vararg{Any}}) where I<:Union{AbstractUnitRange{Int},Int,AbstractCartesianIndex} =
    _row_subarraylayout(par, RowMajor(), tail(inds))
_row_subarraylayout(par::AbstractRowMajor, ::AbstractStridedLayout, inds::Tuple{I, Vararg{Any}}) where I<:Union{RangeIndex,AbstractCartesianIndex} =
    _row_subarraylayout(par, StridedLayout(), tail(inds))

_strided_subarraylayout(par, inds) = UnknownLayout()
_strided_subarraylayout(par, ::Tuple{}) = StridedLayout()
_strided_subarraylayout(par, inds::Tuple{I, Vararg{Any}}) where I<:Union{RangeIndex,AbstractCartesianIndex} =
    _strided_subarraylayout(par, tail(inds))

# MemoryLayout of transposed and adjoint matrices
struct ConjLayout{ML<:MemoryLayout} <: MemoryLayout
    layout::ML
end

conjlayout(_1, _2) = UnknownLayout()
conjlayout(::Type{<:Complex}, M::ConjLayout) = M.layout
conjlayout(::Type{<:Complex}, M::AbstractStridedLayout) = ConjLayout(M)
conjlayout(::Type{<:Real}, M::MemoryLayout) = M


subarraylayout(M::ConjLayout, t::Tuple) = ConjLayout(subarraylayout(M.layout, t))

MemoryLayout(A::Transpose) = transposelayout(MemoryLayout(parent(A)))
MemoryLayout(A::Adjoint) = adjointlayout(eltype(A), MemoryLayout(parent(A)))
transposelayout(_) = UnknownLayout()
transposelayout(::StridedLayout) = StridedLayout()
transposelayout(::ColumnMajor) = RowMajor()
transposelayout(::RowMajor) = ColumnMajor()
transposelayout(::DenseColumnMajor) = DenseRowMajor()
transposelayout(::DenseRowMajor) = DenseColumnMajor()
transposelayout(M::ConjLayout) = ConjLayout(transposelayout(M.layout))
adjointlayout(::Type{T}, M::MemoryLayout) where T = transposelayout(conjlayout(T, M))


# MemoryLayout of Symmetric/Hermitian
"""
    SymmetricLayout(layout, uplo)


is returned by `MemoryLayout(A)` if a matrix `A` has storage in memory
as a symmetrized version of `layout`, where the entries used are dictated by the
`uplo`, which can be `'U'` or `L'`.

A matrix that has memory layout `SymmetricLayout(layout, uplo)` must overrided
`symmetricdata(A)` to return a matrix `B` such that `MemoryLayout(B) == layout` and
`A[k,j] == B[k,j]` for `j ≥ k` if `uplo == 'U'` (`j ≤ k` if `uplo == 'L'`) and
`A[k,j] == B[j,k]` for `j < k` if `uplo == 'U'` (`j > k` if `uplo == 'L'`).
"""
struct SymmetricLayout{ML<:MemoryLayout} <: MemoryLayout
    layout::ML
    uplo::Char
end
SymmetricLayout(layout::ML, uplo) where ML<:MemoryLayout = SymmetricLayout{ML}(layout, uplo)

"""
    HermitianLayout(layout, uplo)


is returned by `MemoryLayout(A)` if a matrix `A` has storage in memory
as a hermitianized version of `layout`, where the entries used are dictated by the
`uplo`, which can be `'U'` or `L'`.

A matrix that has memory layout `HermitianLayout(layout, uplo)` must overrided
`hermitiandata(A)` to return a matrix `B` such that `MemoryLayout(B) == layout` and
`A[k,j] == B[k,j]` for `j ≥ k` if `uplo == 'U'` (`j ≤ k` if `uplo == 'L'`) and
`A[k,j] == conj(B[j,k])` for `j < k` if `uplo == 'U'` (`j > k` if `uplo == 'L'`).
"""
struct HermitianLayout{ML<:MemoryLayout} <: MemoryLayout
    layout::ML
    uplo::Char
end
HermitianLayout(layout::ML, uplo) where ML<:MemoryLayout = HermitianLayout{ML}(layout, uplo)

MemoryLayout(A::Hermitian) = hermitianlayout(eltype(A), MemoryLayout(parent(A)), A.uplo)
MemoryLayout(A::Symmetric) = symmetriclayout(MemoryLayout(parent(A)), A.uplo)
hermitianlayout(_1, _2, _3) = UnknownLayout()
hermitianlayout(::Type{<:Complex}, layout::AbstractColumnMajor, uplo) = HermitianLayout(layout,uplo)
hermitianlayout(::Type{<:Real}, layout::AbstractColumnMajor, uplo) = SymmetricLayout(layout,uplo)
hermitianlayout(::Type{<:Complex}, layout::AbstractRowMajor, uplo) = HermitianLayout(layout,uplo)
hermitianlayout(::Type{<:Real}, layout::AbstractRowMajor, uplo) = SymmetricLayout(layout,uplo)
symmetriclayout(_1, _2) = UnknownLayout()
symmetriclayout(layout::AbstractColumnMajor, uplo) = SymmetricLayout(layout,uplo)
symmetriclayout(layout::AbstractRowMajor, uplo) = SymmetricLayout(layout,uplo)
transposelayout(S::SymmetricLayout) = S
adjointlayout(::Type{T}, S::SymmetricLayout) where T<:Real = S
adjointlayout(::Type{T}, S::HermitianLayout) where T = S
subarraylayout(S::SymmetricLayout, ::Tuple{<:Slice,<:Slice}) = S
subarraylayout(S::HermitianLayout, ::Tuple{<:Slice,<:Slice}) = S

symmetricdata(A::Symmetric) = A.data
symmetricdata(A::Hermitian{<:Real}) = A.data
symmetricdata(V::SubArray{<:Any, 2, <:Any, <:Tuple{<:Slice,<:Slice}}) = symmetricdata(parent(V))
symmetricdata(V::Adjoint{<:Real}) = symmetricdata(parent(V))
symmetricdata(V::Transpose) = symmetricdata(parent(V))
hermitiandata(A::Hermitian) = A.data
hermitiandata(V::SubArray{<:Any, 2, <:Any, <:Tuple{<:Slice,<:Slice}}) = hermitiandata(parent(V))
hermitiandata(V::Adjoint) = hermitiandata(parent(V))
hermitiandata(V::Transpose{<:Real}) = hermitiandata(parent(V))


# MemoryLayout of triangular matrices
struct TriangularLayout{UPLO,UNIT,ML} <: MemoryLayout
    layout::ML
end


TriangularLayout{UPLO,UNIT}(lay) where {UPLO,UNIT} = TriangularLayout{UPLO,UNIT,typeof(lay)}(lay)


"""
    LowerTriangularLayout(layout)

is returned by `MemoryLayout(A)` if a matrix `A` has storage in memory
equivalent to a `LowerTriangular(B)` where `B` satisfies `MemoryLayout(B) == layout`.

A matrix that has memory layout `LowerTriangularLayout(layout)` must overrided
`triangulardata(A)` to return a matrix `B` such that `MemoryLayout(B) == layout` and
`A[k,j] ≡ zero(eltype(A))` for `j > k` and
`A[k,j] ≡ B[k,j]` for `j ≤ k`.

Moreover, `transpose(A)` and `adjoint(A)` must return a matrix that has memory
layout `UpperTriangularLayout`.
"""
const LowerTriangularLayout{ML} = TriangularLayout{'L','N',ML}

"""
    UnitLowerTriangularLayout(ML::MemoryLayout)

is returned by `MemoryLayout(A)` if a matrix `A` has storage in memory
equivalent to a `UnitLowerTriangular(B)` where `B` satisfies `MemoryLayout(B) == layout`.

A matrix that has memory layout `UnitLowerTriangularLayout(layout)` must overrided
`triangulardata(A)` to return a matrix `B` such that `MemoryLayout(B) == layout` and
`A[k,j] ≡ zero(eltype(A))` for `j > k`,
`A[k,j] ≡ one(eltype(A))` for `j == k`,
`A[k,j] ≡ B[k,j]` for `j < k`.

Moreover, `transpose(A)` and `adjoint(A)` must return a matrix that has memory
layout `UnitUpperTriangularLayout`.
"""
const UnitLowerTriangularLayout{ML} = TriangularLayout{'L','U',ML}

"""
    UpperTriangularLayout(ML::MemoryLayout)

is returned by `MemoryLayout(A)` if a matrix `A` has storage in memory
equivalent to a `UpperTriangularLayout(B)` where `B` satisfies `MemoryLayout(B) == ML`.

A matrix that has memory layout `UpperTriangularLayout(layout)` must overrided
`triangulardata(A)` to return a matrix `B` such that `MemoryLayout(B) == layout` and
`A[k,j] ≡ B[k,j]` for `j ≥ k` and
`A[k,j] ≡ zero(eltype(A))` for `j < k`.

Moreover, `transpose(A)` and `adjoint(A)` must return a matrix that has memory
layout `LowerTriangularLayout`.
"""
const UpperTriangularLayout{ML} = TriangularLayout{'U','N',ML}

"""
    UnitUpperTriangularLayout(ML::MemoryLayout)

is returned by `MemoryLayout(A)` if a matrix `A` has storage in memory
equivalent to a `UpperTriangularLayout(B)` where `B` satisfies `MemoryLayout(B) == ML`.

A matrix that has memory layout `UnitUpperTriangularLayout(layout)` must overrided
`triangulardata(A)` to return a matrix `B` such that `MemoryLayout(B) == layout` and
`A[k,j] ≡ B[k,j]` for `j > k`,
`A[k,j] ≡ one(eltype(A))` for `j == k`,
`A[k,j] ≡ zero(eltype(A))` for `j < k`.

Moreover, `transpose(A)` and `adjoint(A)` must return a matrix that has memory
layout `UnitLowerTriangularLayout`.
"""
UnitUpperTriangularLayout{ML} = TriangularLayout{'U','U',ML}


MemoryLayout(A::UpperTriangular) = triangularlayout(UpperTriangularLayout, MemoryLayout(parent(A)))
MemoryLayout(A::UnitUpperTriangular) = triangularlayout(UnitUpperTriangularLayout, MemoryLayout(parent(A)))
MemoryLayout(A::LowerTriangular) = triangularlayout(LowerTriangularLayout, MemoryLayout(parent(A)))
MemoryLayout(A::UnitLowerTriangular) = triangularlayout(UnitLowerTriangularLayout, MemoryLayout(parent(A)))
triangularlayout(_, ::MemoryLayout) = UnknownLayout()
triangularlayout(::Type{Tri}, ML::AbstractColumnMajor) where {Tri} = Tri(ML)
triangularlayout(::Type{Tri}, ML::AbstractRowMajor) where {Tri} = Tri(ML)
triangularlayout(::Type{Tri}, ML::ConjLayout{<:AbstractRowMajor}) where {Tri} = Tri(ML)
subarraylayout(layout::TriangularLayout, ::Tuple{<:Union{Slice,Base.OneTo},<:Union{Slice,Base.OneTo}}) = layout
conjlayout(::Type{<:Complex}, ml::TriangularLayout{UPLO,UNIT}) where {UPLO,UNIT} =
    TriangularLayout{UPLO,UNIT}(ConjLayout(ml.layout))

for (TriLayout, TriLayoutTrans) in ((UpperTriangularLayout,     LowerTriangularLayout),
                                    (UnitUpperTriangularLayout, UnitLowerTriangularLayout),
                                    (LowerTriangularLayout,     UpperTriangularLayout),
                                    (UnitLowerTriangularLayout, UnitUpperTriangularLayout))
    @eval transposelayout(ml::$TriLayout) = $TriLayoutTrans(transposelayout(ml.layout))
end

triangulardata(A::AbstractTriangular) = parent(A)
triangulardata(A::Adjoint) = Adjoint(triangulardata(parent(A)))
triangulardata(A::Transpose) = Transpose(triangulardata(parent(A)))
triangulardata(A::SubArray{<:Any,2,<:Any,<:Tuple{<:Union{Slice,Base.OneTo},<:Union{Slice,Base.OneTo}}}) =
    view(triangulardata(parent(A)), parentindices(A)...)


abstract type AbstractBandedLayout <: MemoryLayout end

struct DiagonalLayout{ML} <: AbstractBandedLayout
    layout::ML
end

struct SymTridiagonalLayout{ML} <: AbstractBandedLayout
    layout::ML
end

MemoryLayout(D::Diagonal) = DiagonalLayout(MemoryLayout(parent(D)))
diagonaldata(D::Diagonal) = parent(D)

MemoryLayout(D::SymTridiagonal) = SymTridiagonalLayout(MemoryLayout(D.dv))
diagonaldata(D::SymTridiagonal) = D.dv
offdiagonaldata(D::SymTridiagonal) = D.ev

transposelayout(ml::DiagonalLayout) = ml
transposelayout(ml::SymTridiagonalLayout) = ml

adjointlayout(_, ml::DiagonalLayout) = ml
adjointlayout(::Type{<:Real}, ml::SymTridiagonal) = ml


###
# Fill
####
abstract type AbstractFillLayout <: MemoryLayout end
struct FillLayout <: AbstractFillLayout end
struct ZerosLayout <: AbstractFillLayout end

MemoryLayout(::AbstractFill) = FillLayout()
MemoryLayout(::Zeros) = ZerosLayout()
