"""
    MulStyle

indicates that an `Applied` object should be materialized
via `Mul`.
"""
struct MulStyle <: AbstractArrayApplyStyle end

Mul(M::Applied) = Mul(M.args...)
arguments(M::Mul) = (M.A, M.B)

similar(M::Applied{MulStyle}, ::Type{T}) where T = similar(Mul(M), T)
copy(M::Applied{MulStyle}) = mul(arguments(M)...)
@inline copyto!(dest::AbstractArray, M::Applied{MulStyle}) = copyto!(dest, Mul(M))

const MulArray{T, N, Args} = ApplyArray{T, N, typeof(*), Args}

const MulVector{T, Args} = MulArray{T, 1, Args}
const MulMatrix{T, Args} = MulArray{T, 2, Args}


check_applied_axes(A::Applied{<:Any,typeof(*)}) = check_mul_axes(A.args...)

size(M::Applied{<:Any,typeof(*)}, p::Int) = size(M)[p]
axes(M::Applied{<:Any,typeof(*)}, p::Int) = axes(M)[p]
ndims(M::Applied{<:Any,typeof(*)}) = ndims(last(M.args))

_mul_ndims(::Type{Tuple{A}}) where A = ndims(A)
_mul_ndims(::Type{Tuple{A,B}}) where {A,B} = ndims(B)
ndims(::Type{<:Applied{<:Any,typeof(*),Args}}) where Args = _mul_ndims(Args)


length(M::Applied{<:Any,typeof(*)}) = prod(size(M))
size(M::Applied{<:Any,typeof(*)}) = length.(axes(M))


@inline _eltypes() = tuple()
@inline _eltypes(A, B...) = tuple(eltype(A), _eltypes(B...)...)

@inline eltype(M::Applied{<:Any,typeof(*)}) = _mul_eltype(_eltypes(M.args...)...)

@inline mulaxes1(::Tuple{}) = ()
@inline mulaxes1(::Tuple{}, B, C...) = mulaxes1(B, C...)
@inline mulaxes1(A::Tuple, C...) = first(A)
@inline mulaxes2(::Tuple{}) = ()
@inline mulaxes2(::Tuple{}, B, C...) = mulaxes2(B, C...)
@inline mulaxes2(A::Tuple{<:Any}, C...) = ()
@inline mulaxes2(A::Tuple{<:Any,<:Any}, C...) = last(A)

@inline _combine_axes(::Tuple{}, ::Tuple{}) = ()
@inline _combine_axes(a, ::Tuple{}) = (a,)
@inline _combine_axes(a, b) = (a,b)
@inline mulaxes(ax...) = _combine_axes(mulaxes1(ax...), mulaxes2(reverse(ax)...))

@inline axes(M::Applied{<:Any,typeof(*)}) = mulaxes(map(axes,M.args)...)
@inline axes(M::Applied{<:Any, typeof(*), Tuple{}}) = ()

###
# show
###

function _applyarray_summary(io::IO, ::typeof(*), args)
    print(io, "(")
    summary(io, first(args))
    print(io, ")")
    for a in tail(args)
        print(io, " * (")
        summary(io, a)
        print(io, ")")
    end
end
    

####
# Matrix * Array
####

combine_mul_styles(a) = a
combine_mul_styles(a, b) = error("Overload for $a and $b")
combine_mul_styles(::T, ::T) where T = T()
combine_mul_styles(::MulStyle, ::MulStyle) = DefaultArrayApplyStyle()
combine_mul_styles(::MulStyle, ::DefaultArrayApplyStyle) = DefaultArrayApplyStyle()
combine_mul_styles(::DefaultArrayApplyStyle, ::MulStyle) = DefaultArrayApplyStyle()
combine_mul_styles(::DefaultArrayApplyStyle, ::DefaultApplyStyle) = DefaultApplyStyle()
combine_mul_styles(::DefaultApplyStyle, ::DefaultArrayApplyStyle) = DefaultApplyStyle()
combine_mul_styles(a, b, c...) = combine_mul_styles(combine_mul_styles(a, b), c...)
# We need to combine all branches to determine whether it can be  simplified
ApplyStyle(::typeof(*), a) = DefaultApplyStyle()
ApplyStyle(::typeof(*), a::AbstractArray) = DefaultArrayApplyStyle()
_mul_ApplyStyle(a, b...) = combine_mul_styles(ApplyStyle(*, a, most(b)...),  ApplyStyle(*, b...))
ApplyStyle(::typeof(*), a, b...) = _mul_ApplyStyle(a, b...)
ApplyStyle(::typeof(*), a::Type{<:AbstractArray}, b::Type{<:AbstractArray}...) = _mul_ApplyStyle(a, b...)
ApplyStyle(::typeof(*), ::Type{<:AbstractArray}) = DefaultArrayApplyStyle()
ApplyStyle(::typeof(*), ::Type{<:Number}, ::Type{<:AbstractArray}) = DefaultArrayApplyStyle()
ApplyStyle(::typeof(*), ::Type{<:AbstractArray}, ::Type{<:Number}) = DefaultArrayApplyStyle()
ApplyStyle(::typeof(*), ::Type{<:AbstractArray}, ::Type{<:AbstractArray}) = MulStyle()


# arguments for something that is a *
@inline _mul_arguments(::ApplyLayout{typeof(*)}, A) = arguments(A)
@inline _mul_arguments(_, A) = (A,)
@inline _mul_arguments(A) = _mul_arguments(MemoryLayout(typeof(A)), A)

@inline __flatten(A::Tuple{<:Any}, B::Tuple) = (A..., _flatten(B...)...)
@inline __flatten(A::Tuple, B::Tuple) = _flatten(A..., B...)
@inline __flatten(A::Tuple{<:Any}, ::Tuple{}) = A
@inline __flatten(A::Tuple, B::Tuple{}) = _flatten(A...)

@inline _flatten() = ()
@inline _flatten(A, B...) = __flatten(_mul_arguments(A), B)
@inline _flatten(A) = __flatten(_mul_arguments(A), ())
@inline flatten(A) = _mul(_flatten(_mul_arguments(A)...)...)





####
# MulArray
#####

_mul(A) = A
_mul(A,B,C...) = lazymaterialize(*,A,B,C...)

_mul_colsupport(j, Z) = colsupport(Z,j)
_mul_colsupport(j, Z::AbstractArray) = colsupport(Z,j)
_mul_colsupport(j, Z, Y...) = axes(Z,1) # default is return all

_mul_colsupport(j, Z::AbstractArray, Y...) = _mul_colsupport(colsupport(Z,j), Y...)

colsupport(B::Applied{<:Any,typeof(*)}, j) = _mul_colsupport(j, reverse(B.args)...)
colsupport(B::MulArray, j) = _mul_colsupport(j, reverse(B.args)...)

_mul_rowsupport(j, A) = rowsupport(A,j)
_mul_rowsupport(j, A::AbstractArray) = rowsupport(A,j)
_mul_rowsupport(j, A, B...) = axes(A,1) # default is return all
_mul_rowsupport(j, A::AbstractArray, B...) = _mul_rowsupport(rowsupport(A,j), B...)

rowsupport(B::Applied{<:Any,typeof(*)}, j) = _mul_rowsupport(j, B.args...)
rowsupport(B::MulArray, j) = _mul_rowsupport(j, B.args...)

function getindex(M::Applied{<:Any,typeof(*)}, k...)
    A,Bs = first(M.args), tail(M.args)
    B = _mul(Bs...)
    Mul(A, B)[k...]
end

_flatten(A::MulArray, B...) = _flatten(Applied(A), B...)
flatten(A::MulArray) = ApplyArray(flatten(Applied(A)))	
 
adjoint(A::MulMatrix) = ApplyArray(*, reverse(map(adjoint,A.args))...)
transpose(A::MulMatrix) = ApplyArray(*, reverse(map(transpose,A.args))...)

###
# sub materialize
###

# determine rows/cols of multiplication
_mul_args_rowsupport(a,kr) = rowsupport(a,kr)
_mul_args_colsupport(a,kr) = colsupport(a,kr)
__mul_args_rows(kr, a) = (kr,)
__mul_args_rows(kr, a, b...) = 
    (kr, __mul_args_rows(_mul_args_rowsupport(a,kr), b...)...)
_mul_args_rows(kr, a, b...) = __mul_args_rows(_mul_args_rowsupport(a,kr), b...)
__mul_args_cols(jr, z) = (jr,)
__mul_args_cols(jr, z, y...) = 
    (__mul_args_cols(_mul_args_colsupport(z,jr), y...)..., jr)
_mul_args_cols(jr, z, y...) = __mul_args_cols(_mul_args_colsupport(z,jr), y...)

sublayout(::ApplyLayout{typeof(*)}, _...) = ApplyLayout{typeof(*)}()
# matrix-indexing loses the multiplication structure as we don't support tensor multiplication
sublayout(::ApplyLayout{typeof(*)}, ::Type{<:Tuple{AbstractMatrix}}) = UnknownLayout()

call(::ApplyLayout{typeof(*)}, V::SubArray) = *

function _mat_mul_arguments(args, (kr,jr)::Tuple{Any,Any})
    kjr = intersect.(_mul_args_rows(kr, args...), _mul_args_cols(jr, reverse(args)...))
    map(view, args, (kr, kjr...), (kjr..., jr))
end

_vec_mul_view(a...) = view(a...)
_vec_mul_view(a::AbstractVector, kr, ::Colon) = view(a, kr)

# this is a vector view of a MulVector
function _vec_mul_arguments(args, (kr,))
    kjr = intersect.(_mul_args_rows(kr, args...), _mul_args_cols(Base.OneTo(1), reverse(args)...))
    _vec_mul_view.(args, (kr, kjr...), (kjr..., :))
end

# this is a vector view of a MulMatrix
_vec_mul_arguments(args, (kr,jr)::Tuple{AbstractVector,Number}) =
    _mat_mul_arguments(args, (kr,jr))

# this is a row-vector view
_vec_mul_arguments(args, (kr,jr)::Tuple{Number,AbstractVector}) =
    _vec_mul_arguments(reverse(map(permutedims, args)), (jr,kr))

_mat_mul_arguments(V) = _mat_mul_arguments(arguments(parent(V)), parentindices(V))
_vec_mul_arguments(V) = _vec_mul_arguments(arguments(parent(V)), parentindices(V))

arguments(::ApplyLayout{typeof(*)}, V::SubArray{<:Any,2}) = _mat_mul_arguments(V)
arguments(::ApplyLayout{typeof(*)}, V::SubArray{<:Any,1}) = _vec_mul_arguments(V)

@inline sub_materialize(::ApplyLayout{typeof(*)}, V) = apply(*, arguments(V)...)


##
# adjoint Mul
##

adjointlayout(::Type, ::ApplyLayout{typeof(*)}) = ApplyLayout{typeof(*)}()
transposelayout(::ApplyLayout{typeof(*)}) = ApplyLayout{typeof(*)}()

call(::ApplyLayout{typeof(*)}, V::Adjoint) = *
call(::ApplyLayout{typeof(*)}, V::Transpose) = *

arguments(::ApplyLayout{typeof(*)}, V::Adjoint) = reverse(adjoint.(arguments(V')))
arguments(::ApplyLayout{typeof(*)}, V::Transpose) = reverse(transpose.(arguments(V')))

permutedims(A::ApplyArray{<:Any,2,typeof(*)}) = ApplyArray(*, reverse(map(permutedims, A.args))...)


##
# * specialcase
##

for op in (:*, :\)
    @eval broadcasted(::DefaultArrayStyle{N}, ::typeof($op), a::Number, b::ApplyArray{<:Number,N,typeof(*)}) where N =
        ApplyArray(*, broadcast($op,a,first(b.args)), tail(b.args)...)
end

broadcasted(::DefaultArrayStyle{N}, ::typeof(/), b::ApplyArray{<:Number,N,typeof(*)}, a::Number) where N =
        ApplyArray(*, most(b.args)..., broadcast(/,last(b.args),a))

for Typ in (:Lmul, :Rmul)
    @eval $Typ(M::Applied{<:Any,typeof(*)}) = $Typ(M.args...)
end



##
# L/Rmul
##

@inline ApplyArray(M::Lmul) = ApplyArray(*, M.A, M.B)
@inline ApplyArray(M::Rmul) = ApplyArray(*, M.A, M.B)
@inline ApplyArray(M::Mul) = ApplyArray(*, M.A, M.B)
@inline ApplyArray(M::Mul{ApplyLayout{typeof(*)},ApplyLayout{typeof(*)}}) = ApplyArray(*, arguments(M.A)..., arguments(M.B)...)
@inline ApplyArray(M::Mul{ApplyLayout{typeof(*)}}) = ApplyArray(*, arguments(M.A)..., M.B)
@inline ApplyArray(M::Mul{<:Any,ApplyLayout{typeof(*)}}) = ApplyArray(*, M.A, arguments(M.B)...)

# Support QuasiArrays

lazymaterialize(::typeof(*), a::AbstractArray) = a
lazymaterialize(F::Function, args::AbstractArray...) = copy(ApplyArray(F, args...))
lazymaterialize(M::Mul) = lazymaterialize(*, M.A, M.B)


### 
# Simplify
# Here we implement a simple routine for simplifying multiplication by expanding what can be expanded

#
# The method is given *(a, b, ..., y, z) for see if *(a, b, ..., y) can be simplified.
# If so, simplify and start over. If not, see if *(b, ..., y, z) can  be simplified. If so,
# simplify and start over. Of not, return a lazy version.
# In ContinuumArrays we use this for simplifying differential operators
###

@inline _or(::Val{true}, ::Val{true}) = Val(true)
@inline _or(::Val{true}, ::Val{false}) = Val(true)
@inline _or(::Val{false}, ::Val{true}) = Val(true)
@inline _or(::Val{false}, ::Val{false}) = Val(false)

@inline _not(::Val{true}) = Val(false)
@inline _not(::Val{false}) = Val(true)

@inline _islazy(::AbstractLazyLayout) = Val(true)
@inline _islazy(_) = Val(false)
@inline islazy(A) = _islazy(MemoryLayout(A))
@inline simplifiable(M::Mul) = _not(_or(islazy(M.A), islazy(M.B)))

@inline simplifiable(::typeof(*), a) = Val(false)
@inline simplifiable(::typeof(*), a, b) = simplifiable(Mul(a,b))
@inline simplifiable(::typeof(*), a...) = _most_simplifiable(*, simplifiable(*, most(a)...), a)
@inline _most_simplifiable(::typeof(*), ::Val{true}, a) = Val(true)
@inline _most_simplifiable(::typeof(*), ::Val{false}, a) = simplifiable(*, tail(a)...)

# Flatten first
@inline simplify(::typeof(*), args...) = _simplify(*, _flatten(args...)...)
@inline _simplify(::typeof(*), a, b) = _twoarg_simplify(*, simplifiable(*, a, b), a, b)
@inline _twoarg_simplify(::typeof(*), ::Val{false}, a, b) = lazymaterialize(*, a, b)
@inline _twoarg_simplify(::typeof(*), ::Val{true}, a, b) = mul(a,b)
@inline _simplify(::typeof(*), args...) = _most_simplify(simplifiable(*, most(args)...), args)
@inline _most_simplify(::Val{true}, args) = *(_mul_arguments(_simplify(*, most(args)...))..., last(args))
@inline _most_simplify(::Val{false}, args) = _tail_simplify(simplifiable(*, tail(args)...), args)
@inline _tail_simplify(::Val{true}, args) = *(first(args), _mul_arguments(_simplify(*, tail(args)...))...)
@inline _tail_simplify(::Val{false}, args) = lazymaterialize(*, args...)

simplify(M::Mul) = simplify(*, M.A, M.B)
simplify(M::Applied{<:Any,typeof(*)}) = simplify(*, arguments(M)...)


@inline copy(M::Mul{<:AbstractLazyLayout,<:AbstractLazyLayout}) = simplify(M)
@inline copy(M::Mul{<:AbstractLazyLayout}) = simplify(M)
@inline copy(M::Mul{<:Any,<:AbstractLazyLayout}) = simplify(M)
@inline copy(M::Mul{<:DualLayout,<:AbstractLazyLayout,<:AbstractMatrix,<:AbstractVector}) = copy(Dot(M))
@inline copy(M::Mul{<:AbstractQLayout,<:AbstractLazyLayout}) = simplify(M)
@inline copy(M::Mul{<:AbstractLazyLayout,<:AbstractQLayout}) = simplify(M)


#TODO: Why not all DiagonalLayout?
@inline copy(M::Mul{<:DiagonalLayout{<:AbstractFillLayout},<:AbstractLazyLayout}) = copy(mulreduce(M))
@inline copy(M::Mul{<:AbstractLazyLayout,<:DiagonalLayout{<:AbstractFillLayout}}) = copy(mulreduce(M))

# inv

function _inv(Lay::ApplyLayout{typeof(*)}, _, A)
    args = arguments(Lay, A)
    map(checksquare,args)
    *(reverse(map(inv, arguments(Lay, A)))...)
end


##
# getindex
##
_reverse_mul_vec(z) = z
_reverse_mul_vec(z, y, w...) = _reverse_mul_vec(y*z, w...)
function getindex(M::ApplyMatrix{<:Any,typeof(*)}, ::Colon, j::Integer)
    rargs = reverse(M.args)
    _reverse_mul_vec(first(rargs)[:,j], tail(rargs)...)
end