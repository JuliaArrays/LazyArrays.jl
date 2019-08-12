
checkdimensions() = nothing
checkdimensions(_) = nothing

function checkdimensions(A, B, C...)
    axes(A,2) == axes(B,1) || throw(DimensionMismatch(""))
    checkdimensions(B, C...)
end

const Mul{Style, Factors<:Tuple} = Applied{Style, typeof(*), Factors}

Mul(A...) = applied(*, A...)

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

_mul_axes(ax1, ::Tuple{}) = (ax1,)
_mul_axes(ax1, ::Tuple{<:Any}) = (ax1,)
_mul_axes(ax1, (_,ax2)::Tuple{<:Any,<:Any}) = (ax1,ax2)
axes(M::Mul) = _mul_axes(axes(first(M.args),1), axes(last(M.args)))
axes(M::Mul{<:Any, Tuple{}}) = ()


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


"""
   lmaterialize(M::Mul)

materializes arrays iteratively, left-to-right.
"""
lmaterialize(M::Mul) = _lmaterialize(M.args...)

_lmaterialize(A, B) = apply(*,A,B)
_lmaterialize(A, B, C, D...) = _lmaterialize(apply(*,A,B), C, D...)

_flatten() = ()
_flatten(A, B...) = (A, _flatten(B...)...)
_flatten(A::Mul, B...) = _flatten(A.args..., B...)
flatten(A) = A
flatten(A::Mul) = Mul(_flatten(A.args...)...)



rowsupport(_, A, k) = axes(A,2)
""""
    rowsupport(A, k)

gives an iterator containing the possible non-zero entries in the k-th row of A.
"""
rowsupport(A, k) = rowsupport(MemoryLayout(A), A, k)

colsupport(_, A, j) = axes(A,1)

""""
    colsupport(A, j)

gives an iterator containing the possible non-zero entries in the j-th column of A.
"""
colsupport(A, j) = colsupport(MemoryLayout(A), A, j)


rowsupport(::DiagonalLayout, _, k) = k:k
colsupport(::DiagonalLayout, _, j) = j:j

rowsupport(::ZerosLayout, _1, _2) = 1:0
colsupport(::ZerosLayout, _1, _2) = 1:0




####
# MulArray
#####

_mul(A) = A
_mul(A,B,C...) = Mul(A,B,C...)


function getindex(M::Mul, k::Integer)
    A,Bs = first(M.args), tail(M.args)
    B = _mul(Bs...)
    ret = zero(eltype(M))
    for j = rowsupport(A, k)
        ret += A[k,j] * B[j]
    end
    ret
end


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


const MulArray{T, N, MUL<:Mul} = ApplyArray{T, N, MUL}

const MulVector{T, MUL<:Mul} = MulArray{T, 1, MUL}
const MulMatrix{T, MUL<:Mul} = MulArray{T, 2, MUL}

const MulLayout{LAY} = ApplyLayout{typeof(*),LAY}
MulLayout(layouts) = ApplyLayout(*, layouts)


_flatten(A::MulArray, B...) = _flatten(A.applied, B...)
flatten(A::MulArray) = ApplyArray(*, flatten(A.applied))

@propagate_inbounds getindex(A::MulArray, k::Int) = A.applied[k]		
@propagate_inbounds getindex(A::MulArray{T,N}, kj::Vararg{Int,N}) where {T,N} =		
     A.applied[kj...]		
 
*(A::MulArray, B::MulArray) = MulArray(A, B)

adjoint(A::MulArray) = ApplyArray(*, reverse(adjoint.(A.applied.args))...)
transpose(A::MulArray) = ApplyArray(*, reverse(transpose.(A.applied.args))...)
