struct Mul{Styles<:Tuple, Factors<:Tuple}
    styles::Styles
    factors::Factors
end


"""
 Mul(A1, A2, …, AN)

represents lazy multiplication A1*A2*…*AN.
"""
Mul(A...) = Mul(MemoryLayout.(A), A)

const Mul2{StyleA, StyleB, AType, BType} = Mul{<:Tuple{StyleA,StyleB}, <:Tuple{AType,BType}}

_mul_eltype(a) = eltype(a)
_mul_eltype(a, b...) = Base.promote_op(*, eltype(a), _mul_eltype(b...))
eltype(M::Mul) = _mul_eltype(M.factors...)
size(M::Mul, p::Int) = size(M)[p]
axes(M::Mul, p::Int) = axes(M)[p]

length(M::Mul) = prod(size(M))
size(M::Mul) = length.(axes(M))

_mul_axes(ax1, ::Tuple{<:Any}) = (ax1,)
_mul_axes(ax1, (_,ax2)::Tuple{<:Any,<:Any}) = (ax1,ax2)
axes(M::Mul) = _mul_axes(axes(first(M.factors),1), axes(last(M.factors)))


####
# Matrix * Array
####

const ArrayMulArray{styleA, styleB, p, q, T, V} =
    Mul2{styleA, styleB, <:AbstractArray{T,p}, <:AbstractArray{V,q}}

@inline copyto!(dest::AbstractArray, M::Mul) = _copyto!(MemoryLayout(dest), dest, M)
# default to Base mul!
function _copyto!(_, dest::AbstractArray, M::ArrayMulArray)
    A,x = M.factors
    mul!(dest, A, x)
end


####
# Matrix * Vector
####
const MatMulVec{styleA, styleB, T, V} = ArrayMulArray{styleA, styleB, 2, 1, T, V}

function getindex(M::MatMulVec, k::Integer)
    A,B = M.factors
    ret = zero(eltype(M))
    for j = 1:size(A,2)
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
    A,B = M.factors
    ret = zero(eltype(M))
    for ℓ in axes(A,2)
        ret += A[k,ℓ] * B[ℓ,j]
    end
    ret
end

getindex(M::MatMulMat, kj::CartesianIndex{2}) = M[kj[1], kj[2]]
