


const Mul{Style, Factors<:Tuple} = Applied{Style, typeof(*), Factors}

const MulArray{T, N, Args} = ApplyArray{T, N, typeof(*), Args}

const MulVector{T, Args} = MulArray{T, 1, Args}
const MulMatrix{T, Args} = MulArray{T, 2, Args}



Mul(A...) = applied(*, A...)


check_applied_axes(A::Mul) = check_mul_axes(A.args...)

size(M::Mul, p::Int) = size(M)[p]
axes(M::Mul, p::Int) = axes(M)[p]
ndims(M::Mul) = ndims(last(M.args))

_mul_ndims(::Type{Tuple{A}}) where A = ndims(A)
_mul_ndims(::Type{Tuple{A,B}}) where {A,B} = ndims(B)
ndims(::Type{<:Mul{<:Any,Args}}) where Args = _mul_ndims(Args)


length(M::Mul) = prod(size(M))
size(M::Mul) = length.(axes(M))


@inline _eltypes() = tuple()
@inline _eltypes(A, B...) = tuple(eltype(A), _eltypes(B...)...)

@inline eltype(M::Mul) = _mul_eltype(_eltypes(M.args...)...)

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

@inline axes(M::Mul) = mulaxes(map(axes,M.args)...)
@inline axes(M::Mul{<:Any, Tuple{}}) = ()


# *(A::Mul, B::Mul) = apply(*,A.args..., B.args...)
# *(A::Mul, B) = apply(*,A.args..., B)
# *(A, B::Mul) = apply(*,A, B.args...)
⋆(A...) = Mul(A...)

function show(io::IO, A::Mul) 
    if length(A.args) == 0 
        print(io, "⋆()")
        return 
    end
    print(io, first(A.args))
    for a in A.args[2:end]
        print(io, '⋆', a)
    end
end


####
# Matrix * Array
####


ApplyStyle(::typeof(*), args::Type{<:AbstractArray}...) = mulapplystyle(MemoryLayout.(args)...)


"""
   lmaterialize(M::Mul)

materializes arrays iteratively, left-to-right.
"""
lmaterialize(M::Mul) = _lmaterialize(M.args...)

_lmaterialize(A, B) = apply(*,A,B)
_lmaterialize(A, B, C, D...) = _lmaterialize(apply(*,A,B), C, D...)

# arguments for something that is a *
@inline _arguments(::ApplyLayout{typeof(*)}, A) = arguments(A)
@inline _arguments(_, A) = (A,)
@inline _arguments(A) = _arguments(MemoryLayout(typeof(A)), A)

@inline __flatten(A::Tuple{<:Any}, B::Tuple) = (A..., _flatten(B...)...)
@inline __flatten(A::Tuple, B::Tuple) = _flatten(A..., B...)

@inline _flatten() = ()
@inline _flatten(A, B...) = __flatten(_arguments(A), B)
@inline flatten(A) = _mul(_flatten(_arguments(A)...)...)


copy(M::Mul{DefaultArrayApplyStyle,<:Tuple{<:Any,<:Any}}) = copyto!(similar(M), M)
copy(A::Mul{DefaultArrayApplyStyle}) = flatten(lmaterialize(A))

struct FlattenMulStyle <: ApplyStyle end

copy(A::Mul{FlattenMulStyle}) = materialize(flatten(A))



####
# MulArray
#####

_mul(A) = A
_mul(A,B,C...) = Mul(A,B,C...)

_mul_colsupport(j, Z) = colsupport(Z,j)
_mul_colsupport(j, Z::AbstractArray) = colsupport(Z,j)
_mul_colsupport(j, Z, Y...) = axes(Z,1) # default is return all
function _mul_colsupport(j, Z::AbstractArray, Y...)
    rws = colsupport(Z,j)
    a = size(Z,1)+1
    b = 0
    for k in rws 
        cs = _mul_colsupport(k, Y...)
        a = min(a,first(cs))
        b = max(b,last(cs))
    end
    a:b
end

colsupport(B::Mul, j) = _mul_colsupport(j, reverse(B.args)...)
colsupport(B::MulArray, j) = _mul_colsupport(j, reverse(B.args)...)


function _getindex(M::Mul, ::Tuple{<:Any}, k::Integer)
    A,Bs = first(M.args), tail(M.args)
    B = _mul(Bs...)
    ret = zero(eltype(M))
    for j = rowsupport(A, k) ∩ colsupport(B,1)
        ret += A[k,j] * B[j]
    end
    ret
end

_getindex(M::Mul, ax, k::Integer) = M[Base._ind2sub(ax, k)...]
getindex(M::Mul, k::Integer) = _getindex(M, axes(M), k)
@propagate_inbounds getindex(A::Mul{LazyArrayApplyStyle}, k::Integer) = Applied{DefaultArrayApplyStyle}(A)[k]


getindex(M::Mul, k::CartesianIndex{1}) = M[convert(Int, k)]
getindex(M::Mul, kj::CartesianIndex{2}) = M[kj[1], kj[2]]




function getindex(M::Mul, k::Integer, j::Integer)
    A,Bs = first(M.args), tail(M.args)
    B = _mul(Bs...)
    ret = zero(eltype(M))
    @inbounds for ℓ in (rowsupport(A,k) ∩ colsupport(B,j))
        ret += A[k,ℓ] * B[ℓ,j]
    end
    ret
end

@propagate_inbounds getindex(A::Mul{LazyArrayApplyStyle}, k::Integer, j::Integer) = 
    Applied{DefaultArrayApplyStyle}(A)[k,j]

_flatten(A::MulArray, B...) = _flatten(Applied(A), B...)
flatten(A::MulArray) = ApplyArray(flatten(Applied(A)))	
 
adjoint(A::MulArray) = ApplyArray(*, reverse(map(adjoint,A.args))...)
transpose(A::MulArray) = ApplyArray(*, reverse(map(transpose,A.args))...)

###
# sub materialize
###

# determine rows/cols of multiplication
__mul_args_rows(kr, a) = (kr,)
__mul_args_rows(kr, a, b...) = 
    (kr, __mul_args_rows(rowsupport(a,kr), b...)...)
_mul_args_rows(kr, a, b...) = __mul_args_rows(rowsupport(a,kr), b...)
__mul_args_cols(jr, z) = (jr,)
__mul_args_cols(jr, z, y...) = 
    (__mul_args_cols(colsupport(z,jr), y...)..., jr)
_mul_args_cols(jr, z, y...) = __mul_args_cols(colsupport(z,jr), y...)

sublayout(::ApplyLayout{typeof(*)}, _...) = ApplyLayout{typeof(*)}()

call(::ApplyLayout{typeof(*)}, V::SubArray) = *

function _mat_mul_arguments(args, (kr,jr))
    kjr = intersect.(_mul_args_rows(kr, args...), _mul_args_cols(jr, reverse(args)...))
    view.(args, (kr, kjr...), (kjr..., jr))
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
    _vec_mul_arguments(reverse(map(transpose, args)), (jr,kr))

_mat_mul_arguments(V) = _mat_mul_arguments(arguments(parent(V)), parentindices(V))
_vec_mul_arguments(V) = _vec_mul_arguments(arguments(parent(V)), parentindices(V))

arguments(::ApplyLayout{typeof(*)}, V::SubArray{<:Any,2}) = _mat_mul_arguments(V)
arguments(::ApplyLayout{typeof(*)}, V::SubArray{<:Any,1}) = _vec_mul_arguments(V)

@inline sub_materialize(::ApplyLayout{typeof(*)}, V) = apply(*, arguments(V)...)
@inline copyto!(dest::AbstractArray{T,N}, src::SubArray{T,N,<:ApplyArray{T,N,typeof(*)}}) where {T,N} = 
    copyto!(dest, Applied(src))

##
# adoint Mul
##

adjointlayout(::Type, ::ApplyLayout{typeof(*)}) = ApplyLayout{typeof(*)}()
transposelayout(::ApplyLayout{typeof(*)}) = ApplyLayout{typeof(*)}()

call(::ApplyLayout{typeof(*)}, V::Adjoint) = *
call(::ApplyLayout{typeof(*)}, V::Transpose) = *

arguments(::ApplyLayout{typeof(*)}, V::Adjoint) = reverse(adjoint.(arguments(V')))
arguments(::ApplyLayout{typeof(*)}, V::Transpose) = reverse(transpose.(arguments(V')))



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
    @eval $Typ(M::Mul) = $Typ(M.args...)
end



##
# L/Rmul
##

struct LmulStyle <: AbstractArrayApplyStyle end
struct RmulStyle <: AbstractArrayApplyStyle end

similar(M::Applied{LmulStyle}, ::Type{T}) where T = similar(Lmul(M), T)
copy(M::Applied{LmulStyle}) = copy(Lmul(M))

similar(M::Applied{RmulStyle}, ::Type{T}) where T = similar(Rmul(M), T)
copy(M::Applied{RmulStyle}) = copy(Rmul(M))


@inline copyto!(dest::AbstractVecOrMat, M::Mul{LmulStyle}) = copyto!(dest, Lmul(M.args...))
@inline copyto!(dest::AbstractVecOrMat, M::Mul{RmulStyle}) = copyto!(dest, Rmul(M.args...))

@inline materialize!(M::Mul{LmulStyle}) = materialize!(Lmul(M))
@inline materialize!(M::Mul{RmulStyle}) = materialize!(Rmul(M))


mulapplystyle(::QLayout, _) = LmulStyle()
mulapplystyle(::QLayout, ::LazyLayout) = LazyArrayApplyStyle()

factorizestyle(_) = DefaultArrayApplyStyle()

for op in (:factorize, :qr, :lu, :cholesky)
    @eval begin
        $op(B::LazyMatrix) = apply($op, B)
        ApplyStyle(::typeof($op), B::Type{<:AbstractMatrix}) = factorizestyle(MemoryLayout(B))
        materialize(A::Applied{DefaultArrayApplyStyle,typeof($op),<:Tuple{<:AbstractMatrix{T}}}) where T = 
            Base.invoke($op, Tuple{AbstractMatrix{T}}, A.args...)

        eltype(A::Applied{<:Any,typeof($op)}) = float(eltype(first(A.args)))
    end
end
