
tuple_type_reverse(::Type{T}) where T<:Tuple = Tuple{reverse(tuple(T.parameters...))...}
tuple_type_reverse(::Type{Tuple{}}) = Tuple{}
tuple_type_reverse(::Type{Tuple{A}}) where A = Tuple{A}
tuple_type_reverse(::Type{Tuple{A,B}}) where {A,B} = Tuple{B,A}


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
@inline MemoryLayout(::Type) = UnknownLayout()

@inline MemoryLayout(::Type{<:Number}) = ScalarLayout()
@inline MemoryLayout(::Type{<:DenseArray}) = DenseColumnMajor()

@inline MemoryLayout(::Type{<:ReinterpretArray{T,N,S,P}}) where {T,N,S,P} = reinterpretedmemorylayout(MemoryLayout(P))
@inline reinterpretedmemorylayout(::MemoryLayout) = UnknownLayout()
@inline reinterpretedmemorylayout(::DenseColumnMajor) = DenseColumnMajor()

@inline MemoryLayout(A::Type{<:ReshapedArray{T,N,P}}) where {T,N,P} = reshapedmemorylayout(MemoryLayout(P))
@inline reshapedmemorylayout(::MemoryLayout) = UnknownLayout()
@inline reshapedmemorylayout(::DenseColumnMajor) = DenseColumnMajor()


@inline MemoryLayout(A::Type{<:SubArray{T,N,P,I}}) where {T,N,P,I} = 
    subarraylayout(MemoryLayout(P), I)
subarraylayout(_1, _2) = UnknownLayout()
subarraylayout(_1, _2, _3)= UnknownLayout()
subarraylayout(::DenseColumnMajor, ::Type{<:Tuple{<:Union{AbstractUnitRange{Int},Int,AbstractCartesianIndex}}}) =
    DenseColumnMajor()  # A[:] is DenseColumnMajor if A is DenseColumnMajor
subarraylayout(ml::AbstractColumnMajor, inds) = _column_subarraylayout1(ml, inds)
subarraylayout(::AbstractRowMajor, ::Type{<:Tuple{<:Any}}) =
    UnknownLayout()  # A[:] does not have any structure if A is AbstractRowMajor
subarraylayout(ml::AbstractRowMajor, inds) = _row_subarraylayout1(ml, tuple_type_reverse(inds))
subarraylayout(ml::AbstractStridedLayout, inds) = _strided_subarraylayout(ml, inds)

_column_subarraylayout1(::DenseColumnMajor, inds::Type{<:Tuple{I,Vararg{Int}}}) where I<:Union{Int,AbstractCartesianIndex} =
    DenseColumnMajor() # view(A,1,1,2) is a scalar, which we include in DenseColumnMajor
_column_subarraylayout1(::DenseColumnMajor, inds::Type{<:Tuple{I,Vararg{Int}}}) where I<:Slice =
    DenseColumnMajor() # view(A,:,1,2) is a DenseColumnMajor vector
_column_subarraylayout1(::DenseColumnMajor, inds::Type{<:Tuple{I,Vararg{Int}}}) where I<:AbstractUnitRange{Int} =
    DenseColumnMajor() # view(A,1:3,1,2) is a DenseColumnMajor vector
_column_subarraylayout1(par, inds::Type{<:Tuple{I,Vararg{Int}}}) where I<:Union{Int,AbstractCartesianIndex} =
    DenseColumnMajor() # view(A,1,1,2) is a scalar, which we include in DenseColumnMajor
_column_subarraylayout1(par, inds::Type{<:Tuple{I,Vararg{Int}}}) where I<:AbstractUnitRange{Int} =
    DenseColumnMajor() # view(A,1:3,1,2) is a DenseColumnMajor vector
_column_subarraylayout1(::DenseColumnMajor, inds::Type{<:Tuple{I,Vararg{Any}}}) where I<:Slice =
    _column_subarraylayout(DenseColumnMajor(), DenseColumnMajor(), tuple_type_tail(inds))
_column_subarraylayout1(par::DenseColumnMajor, inds::Type{<:Tuple{I,Vararg{Any}}}) where I<:AbstractUnitRange{Int} =
    _column_subarraylayout(par, ColumnMajor(), tuple_type_tail(inds))
_column_subarraylayout1(par, inds::Type{<:Tuple{I,Vararg{Any}}}) where I<:AbstractUnitRange{Int} =
    _column_subarraylayout(par, ColumnMajor(), tuple_type_tail(inds))
_column_subarraylayout1(par::DenseColumnMajor, inds::Type{<:Tuple{I,Vararg{Any}}}) where I<:Union{RangeIndex,AbstractCartesianIndex} =
    _column_subarraylayout(par, StridedLayout(), tuple_type_tail(inds))
_column_subarraylayout1(par, inds::Type{<:Tuple{I,Vararg{Any}}}) where I<:Union{RangeIndex,AbstractCartesianIndex} =
    _column_subarraylayout(par, StridedLayout(), tuple_type_tail(inds))
_column_subarraylayout1(par, inds) = UnknownLayout()
_column_subarraylayout(par, ret, ::Type{<:Tuple{}}) = ret
_column_subarraylayout(par, ret, ::Type{<:Tuple{I}}) where I = UnknownLayout()
_column_subarraylayout(::DenseColumnMajor, ::DenseColumnMajor, inds::Type{<:Tuple{I,Vararg{Int}}}) where I<:Union{AbstractUnitRange{Int},Int,AbstractCartesianIndex} =
    DenseColumnMajor() # A[:,1:3,1,2] is DenseColumnMajor if A is DenseColumnMajor
_column_subarraylayout(par::DenseColumnMajor, ::DenseColumnMajor, inds::Type{<:Tuple{I, Vararg{Int}}}) where I<:Slice =
    DenseColumnMajor()
_column_subarraylayout(par::DenseColumnMajor, ::DenseColumnMajor, inds::Type{<:Tuple{I, Vararg{Any}}}) where I<:Slice =
    _column_subarraylayout(par, DenseColumnMajor(), tuple_type_tail(inds))
_column_subarraylayout(par, ::AbstractColumnMajor, inds::Type{<:Tuple{I, Vararg{Any}}}) where I<:Union{AbstractUnitRange{Int},Int,AbstractCartesianIndex} =
    _column_subarraylayout(par, ColumnMajor(), tuple_type_tail(inds))
_column_subarraylayout(par, ::AbstractStridedLayout, inds::Type{<:Tuple{I, Vararg{Any}}}) where I<:Union{RangeIndex,AbstractCartesianIndex} =
    _column_subarraylayout(par, StridedLayout(), tuple_type_tail(inds))

_row_subarraylayout1(par, inds::Type{<:Tuple{I,Vararg{Int}}}) where I<:Union{Int,AbstractCartesianIndex} =
    DenseColumnMajor() # view(A,1,1,2) is a scalar, which we include in DenseColumnMajor
_row_subarraylayout1(::DenseRowMajor, inds::Type{<:Tuple{I,Vararg{Int}}}) where I<:Slice =
    DenseColumnMajor() # view(A,1,2,:) is a DenseColumnMajor vector
_row_subarraylayout1(par, inds::Type{<:Tuple{I,Vararg{Int}}}) where I<:AbstractUnitRange{Int} =
    DenseColumnMajor() # view(A,1,2,1:3) is a DenseColumnMajor vector
_row_subarraylayout1(::DenseRowMajor, inds::Type{<:Tuple{I,Vararg{Any}}}) where I<:Slice =
    _row_subarraylayout(DenseRowMajor(), DenseRowMajor(), tuple_type_tail(inds))
_row_subarraylayout1(par, inds::Type{<:Tuple{I,Vararg{Any}}}) where I<:AbstractUnitRange{Int} =
    _row_subarraylayout(par, RowMajor(), tuple_type_tail(inds))
_row_subarraylayout1(par, inds::Type{<:Tuple{I,Vararg{Any}}}) where I<:Union{RangeIndex,AbstractCartesianIndex} =
    _row_subarraylayout(par, StridedLayout(), tuple_type_tail(inds))
_row_subarraylayout1(par, inds) = UnknownLayout()
_row_subarraylayout(par, ret, ::Type{<:Tuple{}}) = ret
_row_subarraylayout(par, ret, ::Type{<:Tuple{I}}) where I = UnknownLayout()
_row_subarraylayout(::DenseRowMajor, ::DenseRowMajor, inds::Type{<:Tuple{I,Vararg{Int}}}) where I<:Union{AbstractUnitRange{Int},Int,AbstractCartesianIndex} =
    DenseRowMajor() # A[1,2,1:3,:] is DenseRowMajor if A is DenseRowMajor
_row_subarraylayout(par::DenseRowMajor, ::DenseRowMajor, inds::Type{<:Tuple{I, Vararg{Int}}}) where I<:Slice =
    DenseRowMajor()
_row_subarraylayout(par::DenseRowMajor, ::DenseRowMajor, inds::Type{<:Tuple{I, Vararg{Any}}}) where I<:Slice =
    _row_subarraylayout(par, DenseRowMajor(), tuple_type_tail(inds))
_row_subarraylayout(par::AbstractRowMajor, ::AbstractRowMajor, inds::Type{<:Tuple{I, Vararg{Any}}}) where I<:Union{AbstractUnitRange{Int},Int,AbstractCartesianIndex} =
    _row_subarraylayout(par, RowMajor(), tuple_type_tail(inds))
_row_subarraylayout(par::AbstractRowMajor, ::AbstractStridedLayout, inds::Type{<:Tuple{I, Vararg{Any}}}) where I<:Union{RangeIndex,AbstractCartesianIndex} =
    _row_subarraylayout(par, StridedLayout(), tuple_type_tail(inds))

_strided_subarraylayout(par, inds) = UnknownLayout()
_strided_subarraylayout(par, ::Type{<:Tuple{}}) = StridedLayout()
_strided_subarraylayout(par, inds::Type{<:Tuple{I, Vararg{Any}}}) where I<:Union{RangeIndex,AbstractCartesianIndex} =
    _strided_subarraylayout(par, tuple_type_tail(inds))

# MemoryLayout of transposed and adjoint matrices
struct ConjLayout{ML<:MemoryLayout} <: MemoryLayout end

conjlayout(_1, _2) = UnknownLayout()
conjlayout(::Type{<:Complex}, ::ConjLayout{ML}) where ML = ML()
conjlayout(::Type{<:Complex}, ::ML) where ML<:AbstractStridedLayout = ConjLayout{ML}()
conjlayout(::Type{<:Real}, M::MemoryLayout) = M


subarraylayout(::ConjLayout{ML}, t::Type{<:Tuple}) where ML = ConjLayout{typeof(subarraylayout(ML(), t))}()

MemoryLayout(::Type{Transpose{T,P}}) where {T,P} = transposelayout(MemoryLayout(P))
MemoryLayout(::Type{Adjoint{T,P}}) where {T,P} = adjointlayout(T, MemoryLayout(P))
transposelayout(_) = UnknownLayout()
transposelayout(::StridedLayout) = StridedLayout()
transposelayout(::ColumnMajor) = RowMajor()
transposelayout(::RowMajor) = ColumnMajor()
transposelayout(::DenseColumnMajor) = DenseRowMajor()
transposelayout(::DenseRowMajor) = DenseColumnMajor()
transposelayout(::ConjLayout{ML}) where ML = ConjLayout{typeof(transposelayout(ML()))}()
adjointlayout(::Type{T}, M::MemoryLayout) where T = transposelayout(conjlayout(T, M))


# MemoryLayout of Symmetric/Hermitian
"""
    SymmetricLayout{layout}()


is returned by `MemoryLayout(A)` if a matrix `A` has storage in memory
as a symmetrized version of `layout`, where the entries used are dictated by the
`uplo`, which can be `'U'` or `L'`.

A matrix that has memory layout `SymmetricLayout(layout, uplo)` must overrided
`symmetricdata(A)` to return a matrix `B` such that `MemoryLayout(B) == layout` and
`A[k,j] == B[k,j]` for `j ≥ k` if `uplo == 'U'` (`j ≤ k` if `uplo == 'L'`) and
`A[k,j] == B[j,k]` for `j < k` if `uplo == 'U'` (`j > k` if `uplo == 'L'`).
"""
struct SymmetricLayout{ML<:MemoryLayout} <: MemoryLayout end

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
struct HermitianLayout{ML<:MemoryLayout} <: MemoryLayout end

MemoryLayout(::Type{Hermitian{T,P}}) where {T,P} = hermitianlayout(T, MemoryLayout(P))
MemoryLayout(::Type{Symmetric{T,P}}) where {T,P} = symmetriclayout(MemoryLayout(P))
hermitianlayout(_1, _2) = UnknownLayout()
hermitianlayout(::Type{<:Complex}, ::ML) where ML<:AbstractColumnMajor = HermitianLayout{ML}()
hermitianlayout(::Type{<:Real}, ::ML) where ML<:AbstractColumnMajor = SymmetricLayout{ML}()
hermitianlayout(::Type{<:Complex}, ::ML) where ML<:AbstractRowMajor = HermitianLayout{ML}()
hermitianlayout(::Type{<:Real}, ::ML) where ML<:AbstractRowMajor = SymmetricLayout{ML}()
symmetriclayout(_1) = UnknownLayout()
symmetriclayout(::ML) where ML<:AbstractColumnMajor = SymmetricLayout{ML}()
symmetriclayout(::ML) where ML<:AbstractRowMajor = SymmetricLayout{ML}()
transposelayout(S::SymmetricLayout) = S
adjointlayout(::Type{T}, S::SymmetricLayout) where T<:Real = S
adjointlayout(::Type{T}, S::HermitianLayout) where T = S
subarraylayout(S::SymmetricLayout, ::Type{<:Tuple{<:Slice,<:Slice}}) = S
subarraylayout(S::HermitianLayout, ::Type{<:Tuple{<:Slice,<:Slice}}) = S

symmetricdata(A::Symmetric) = A.data
symmetricdata(A::Hermitian{<:Real}) = A.data
symmetricdata(V::SubArray{<:Any, 2, <:Any, <:Tuple{<:Slice,<:Slice}}) = symmetricdata(parent(V))
symmetricdata(V::Adjoint{<:Real}) = symmetricdata(parent(V))
symmetricdata(V::Transpose) = symmetricdata(parent(V))
hermitiandata(A::Hermitian) = A.data
hermitiandata(V::SubArray{<:Any, 2, <:Any, <:Tuple{<:Slice,<:Slice}}) = hermitiandata(parent(V))
hermitiandata(V::Adjoint) = hermitiandata(parent(V))
hermitiandata(V::Transpose{<:Real}) = hermitiandata(parent(V))

symmetricuplo(A::Symmetric) = A.uplo
symmetricuplo(A::Hermitian) = A.uplo
symmetricuplo(A::SubArray{<:Any, 2, <:Any, <:Tuple{<:Slice,<:Slice}}) = symmetricuplo(parent(A))

# MemoryLayout of triangular matrices
struct TriangularLayout{UPLO,UNIT,ML} <: MemoryLayout end


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


MemoryLayout(A::Type{UpperTriangular{T,P}}) where {T,P} = triangularlayout(UpperTriangularLayout, MemoryLayout(P))
MemoryLayout(A::Type{UnitUpperTriangular{T,P}}) where {T,P} = triangularlayout(UnitUpperTriangularLayout, MemoryLayout(P))
MemoryLayout(A::Type{LowerTriangular{T,P}}) where {T,P} = triangularlayout(LowerTriangularLayout, MemoryLayout(P))
MemoryLayout(A::Type{UnitLowerTriangular{T,P}}) where {T,P} = triangularlayout(UnitLowerTriangularLayout, MemoryLayout(P))
triangularlayout(_, ::MemoryLayout) = UnknownLayout()
triangularlayout(::Type{Tri}, ::ML) where {Tri, ML<:AbstractColumnMajor} = Tri{ML}()
triangularlayout(::Type{Tri}, ::ML) where {Tri, ML<:AbstractRowMajor} = Tri{ML}()
triangularlayout(::Type{Tri}, ::ML) where {Tri, ML<:ConjLayout{<:AbstractRowMajor}} = Tri{ML}()
subarraylayout(layout::TriangularLayout, ::Type{<:Tuple{<:Union{Slice,Base.OneTo},<:Union{Slice,Base.OneTo}}}) = layout
conjlayout(::Type{<:Complex}, ::TriangularLayout{UPLO,UNIT,ML}) where {UPLO,UNIT,ML} =
    TriangularLayout{UPLO,UNIT,ConjLayout{ML}}()

for (TriLayout, TriLayoutTrans) in ((UpperTriangularLayout,     LowerTriangularLayout),
                                    (UnitUpperTriangularLayout, UnitLowerTriangularLayout),
                                    (LowerTriangularLayout,     UpperTriangularLayout),
                                    (UnitLowerTriangularLayout, UnitUpperTriangularLayout))
    @eval transposelayout(::$TriLayout{ML}) where ML = $TriLayoutTrans{typeof(transposelayout(ML()))}()
end

triangulardata(A::AbstractTriangular) = parent(A)
triangulardata(A::Adjoint) = Adjoint(triangulardata(parent(A)))
triangulardata(A::Transpose) = Transpose(triangulardata(parent(A)))
triangulardata(A::SubArray{<:Any,2,<:Any,<:Tuple{<:Union{Slice,Base.OneTo},<:Union{Slice,Base.OneTo}}}) =
    view(triangulardata(parent(A)), parentindices(A)...)


abstract type AbstractBandedLayout <: MemoryLayout end

struct DiagonalLayout{ML} <: AbstractBandedLayout end
struct SymTridiagonalLayout{ML} <: AbstractBandedLayout end

diagonallayout(_) = UnknownLayout()
diagonallayout(::ML) where ML<:AbstractStridedLayout = DiagonalLayout{ML}()
MemoryLayout(D::Type{Diagonal{T,P}}) where {T,P} = diagonallayout(MemoryLayout(P))
diagonaldata(D::Diagonal) = parent(D)

MemoryLayout(::Type{SymTridiagonal{T,P}}) where {T,P} = SymTridiagonalLayout{typeof(MemoryLayout(P))}()
diagonaldata(D::SymTridiagonal) = D.dv
offdiagonaldata(D::SymTridiagonal) = D.ev

transposelayout(ml::DiagonalLayout) = ml
transposelayout(ml::SymTridiagonalLayout) = ml
transposelayout(ml::ConjLayout{DiagonalLayout}) = ml

adjointlayout(::Type{<:Real}, ml::SymTridiagonalLayout) = ml

###
# Fill
####
abstract type AbstractFillLayout <: MemoryLayout end
struct FillLayout <: AbstractFillLayout end
struct ZerosLayout <: AbstractFillLayout end
struct EyeLayout <: MemoryLayout end

MemoryLayout(::Type{<:AbstractFill}) = FillLayout()
MemoryLayout(::Type{<:Zeros}) = ZerosLayout()
diagonallayout(::ML) where ML<:AbstractFillLayout = DiagonalLayout{ML}()