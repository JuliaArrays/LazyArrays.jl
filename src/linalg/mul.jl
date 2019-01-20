
checkdimensions() = nothing
checkdimensions(_) = nothing

function checkdimensions(A, B, C...)
    axes(A,2) == axes(B,1) || throw(DimensionMismatch(""))
    checkdimensions(B, C...)
end

const Mul{Styles<:Tuple, Factors<:Tuple} = Applied{<:LayoutApplyStyle{Styles}, typeof(*), Factors}

ApplyStyle(::typeof(*), args::AbstractArray...) = LayoutApplyStyle(MemoryLayout.(args))

Mul(A...) = applied(*, A...)

const Mul2{StyleA, StyleB, AType, BType} = Mul{<:Tuple{StyleA,StyleB}, <:Tuple{AType,BType}}

size(M::Mul, p::Int) = size(M)[p]
axes(M::Mul, p::Int) = axes(M)[p]
ndims(M::Mul) = ndims(last(M.args))

length(M::Mul) = prod(size(M))
size(M::Mul) = length.(axes(M))
eltype(M::Mul) = Base.promote_op(*, eltype.(M.args)...)

_mul_axes(ax1, ::Tuple{}) = (ax1,)
_mul_axes(ax1, ::Tuple{<:Any}) = (ax1,)
_mul_axes(ax1, (_,ax2)::Tuple{<:Any,<:Any}) = (ax1,ax2)
axes(M::Mul) = _mul_axes(axes(first(M.args),1), axes(last(M.args)))
axes(M::Mul{Tuple{}}) = ()





"""
   lmaterialize(M::Mul)

materializes arrays iteratively, left-to-right.
"""
lmaterialize(M::Mul) = _lmaterialize(M.args...)

_lmaterialize(A, B) = materialize(Mul(A,B))
_lmaterialize(A, B, C, D...) = _lmaterialize(materialize(Mul(A,B)), C, D...)

"""
   rmaterialize(M::Mul)

materializes arrays iteratively, right-to-left.
"""
rmaterialize(M::Mul) = _rmaterialize(reverse(M.args)...)

_rmaterialize(Z, Y) = materialize(Mul(Y,Z))
_rmaterialize(Z, Y, X, W...) = _rmaterialize(materialize(Mul(Y,Z)), X, W...)


# *(A::Mul, B::Mul) = materialize(Mul(A.args..., B.args...))
# *(A::Mul, B) = materialize(Mul(A.args..., B))
# *(A, B::Mul) = materialize(Mul(A, B.args...))
⋆(A...) = Mul(A...)


####
# Matrix * Array
####

const ArrayMulArray{styleA, styleB, p, q, T, V} =
    Mul2{styleA, styleB, <:AbstractArray{T,p}, <:AbstractArray{V,q}}

const ArrayMuls = Mul{<:Tuple, <:Tuple{Vararg{<:AbstractArray}}}

# the default is always Array
similar(M::ArrayMuls, ::Type{T}, ::NTuple{N,OneTo{Int}}) where {T,N} = Array{T}(undef, size(M))
similar(M::ArrayMuls, ::Type{T}) where T = similar(M, T, axes(M))
_materialize(M::ArrayMulArray, _) = copyto!(similar(M), M)
_materialize(M::ArrayMuls, _) = lmaterialize(M)
_materialize(M::Mul, _) = lmaterialize(M)
_materialize(M::Mul2, _) = error("Cannot materialize $M")
materialize(M::Mul) = _materialize(M, axes(M))


####
# Matrix * Vector
####
const MatMulVec{styleA, styleB, T, V} = ArrayMulArray{styleA, styleB, 2, 1, T, V}


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


function getindex(M::MatMulVec, k::Integer)
    A,B = M.args
    ret = zero(eltype(M))
    for j = rowsupport(A, k)
        ret += A[k,j] * B[j]
    end
    ret
end

getindex(M::MatMulVec, k::CartesianIndex{1}) = M[convert(Int, k)]



####
# Matrix * Matrix
####



const MatMulMat{styleA, styleB, T, V} = ArrayMulArray{styleA, styleB, 2, 2, T, V}

function getindex(M::MatMulMat, k::Integer, j::Integer)
    A,B = M.args
    ret = zero(eltype(M))
    @inbounds for ℓ in (rowsupport(A,k) ∩ colsupport(B,j))
        ret += A[k,ℓ] * B[ℓ,j]
    end
    ret
end

getindex(M::MatMulMat, kj::CartesianIndex{2}) = M[kj[1], kj[2]]


####
# MulArray
#####

function getindex(M::Mul, k)
    A,Bs = first(M.args), tail(M.args)
    B = Mul(Bs)
    ret = zero(eltype(M))
    for j = rowsupport(A, k)
        ret += A[k,j] * B[j]
    end
    ret
end

function getindex(M::Mul, k, j)
    A,Bs = first(M.args), tail(M.args)
    B = Mul(Bs)
    ret = zero(eltype(M))
    @inbounds for ℓ in (rowsupport(A,k) ∩ colsupport(B,j))
        ret += A[k,ℓ] * B[ℓ,j]
    end
    ret
end


struct MulArray{T, N, MUL<:Mul} <: AbstractArray{T,N}
    mul::MUL
end

const MulVector{T, MUL<:Mul} = MulArray{T, 1, MUL}
const MulMatrix{T, MUL<:Mul} = MulArray{T, 2, MUL}


MulArray{T,N}(M::MUL) where {T,N,MUL<:Mul} = MulArray{T,N,MUL}(M)
MulArray{T}(M::Mul) where {T} = MulArray{T,ndims(M)}(M)
MulArray(M::Mul) = MulArray{eltype(M)}(M)
MulVector(M::Mul) = MulVector{eltype(M)}(M)
MulMatrix(M::Mul) = MulMatrix{eltype(M)}(M)

MulArray(factors...) = MulArray(Mul(factors...))
MulArray{T}(factors...) where T = MulArray{T}(Mul(factors...))
MulArray{T,N}(factors...) where {T,N} = MulArray{T,N}(Mul(factors...))
MulVector(factors...) = MulVector(Mul(factors...))
MulMatrix(factors...) = MulMatrix(Mul(factors...))

axes(A::MulArray) = axes(A.mul)
size(A::MulArray) = map(length, axes(A))

IndexStyle(::MulArray{<:Any,1}) = IndexLinear()

@propagate_inbounds getindex(A::MulArray, kj::Int...) = A.mul[kj...]

*(A::MulArray, B::MulArray) = A.mul * B.mul
*(A::MulArray, B::Mul) = A.mul * B
*(A::Mul, B::MulArray) = A * B.mul

adjoint(A::MulArray) = MulArray(reverse(adjoint.(A.mul.args))...)
transpose(A::MulArray) = MulArray(reverse(transpose.(A.mul.args))...)


struct MulLayout{LAY} <: MemoryLayout
    layouts::LAY
end

MemoryLayout(M::MulArray) = MulLayout(MemoryLayout.(M.mul.args))


_flatten(A::MulArray, B...) = _flatten(A.mul.args..., B...)
flatten(A::MulArray) = MulArray(Mul(_flatten(A.mul.args...)))
