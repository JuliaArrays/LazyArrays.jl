


const Mul{Style, Factors<:Tuple} = Applied{Style, typeof(*), Factors}

Mul(A...) = applied(*, A...)

check_mul_axes(A) = nothing
_check_mul_axes(::Number, ::Number) = nothing
_check_mul_axes(::Number, _) = nothing
_check_mul_axes(_, ::Number) = nothing
_check_mul_axes(A, B) = axes(A,2) == axes(B,1) || throw(DimensionMismatch("Second axis of A, $(axes(A,2)), and first axis of B, $(axes(B,1)) must match"))
function check_mul_axes(A, B, C...) 
    _check_mul_axes(A, B)
    check_mul_axes(B, C...)
end

check_applied_axes(A::Mul) = check_mul_axes(A.args...)

size(M::Mul, p::Int) = size(M)[p]
axes(M::Mul, p::Int) = axes(M)[p]
ndims(M::Mul) = ndims(last(M.args))

_mul_ndims(::Type{Tuple{A}}) where A = ndims(A)
_mul_ndims(::Type{Tuple{A,B}}) where {A,B} = ndims(B)
ndims(::Type{<:Mul{<:Any,Args}}) where Args = _mul_ndims(Args)


length(M::Mul) = prod(size(M))
size(M::Mul) = length.(axes(M))

@inline _mul_eltype(A) = A
@inline _mul_eltype(A, B) = Base.promote_op(*, A, B)
@inline _mul_eltype(A, B, C, D...) = _mul_eltype(Base.promote_op(*, A, B), C, D...)

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

@inline _flatten() = ()
@inline _flatten(A, B...) = (A, _flatten(B...)...)
@inline _flatten(A::Mul, B...) = _flatten(A.args..., B...)
@inline flatten(A) = A
@inline flatten(A::Mul) = applied(*, _flatten(A.args...)...)

copy(A::Mul{DefaultArrayApplyStyle}) = flatten(lmaterialize(A))


rowsupport(_, A, k) = axes(A,2)
""""
    rowsupport(A, k)

gives an iterator containing the possible non-zero entries in the k-th row of A.
"""
rowsupport(A, k) = rowsupport(MemoryLayout(typeof(A)), A, k)

colsupport(_, A, j) = axes(A,1)

""""
    colsupport(A, j)

gives an iterator containing the possible non-zero entries in the j-th column of A.
"""
colsupport(A, j) = colsupport(MemoryLayout(typeof(A)), A, j)

rowsupport(::Diagonal, k) = k:k
colsupport(::Diagonal, j) = j:j

rowsupport(::DiagonalLayout, _, k) = k:k
colsupport(::DiagonalLayout, _, j) = j:j

rowsupport(::ZerosLayout, _1, _2) = 1:0
colsupport(::ZerosLayout, _1, _2) = 1:0




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
    a = 1
    b = 0
    for k in rws 
        cs = _mul_colsupport(k, Y...)
        a = min(a,first(cs))
        b = max(b,last(cs))
    end
    a:b
end

colsupport(B::Mul, j) = _mul_colsupport(j, reverse(B.args)...)
colsupport(A::ApplyArray, j) = colsupport(Applied(A), j)


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


const MulArray{T, N, Args} = ApplyArray{T, N, typeof(*), Args}

const MulVector{T, Args} = MulArray{T, 1, Args}
const MulMatrix{T, Args} = MulArray{T, 2, Args}

const MulLayout{LAY} = ApplyLayout{typeof(*),LAY}
MulLayout(layouts) = ApplyLayout(*, layouts)


_flatten(A::MulArray, B...) = _flatten(Applied(A), B...)
flatten(A::MulArray) = ApplyArray(flatten(Applied(A)))	
 
*(A::MulMatrix, B::MulMatrix) = ApplyArray(*, A.args..., B.args...)
*(A::MulMatrix, B::MulVector) = ApplyArray(*, A.args..., B.args...)
*(A::MulVector, B::MulMatrix) = ApplyArray(*, A.args..., B.args...)

adjoint(A::MulArray) = ApplyArray(*, reverse(map(adjoint,A.args))...)
transpose(A::MulArray) = ApplyArray(*, reverse(map(transpose,A.args))...)


