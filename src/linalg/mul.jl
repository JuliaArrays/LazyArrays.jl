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

length(M::MatMulVec) = size(first(M.factors),1)
axes(M::MatMulVec) = (axes(first(M.factors),1),)

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

MatMulMat{styleA, styleB, T, V} = ArrayMulArray{styleA, styleB, 2, 2, T, V}



size(M::MatMulMat) = size.(M.factors,(1,2))
axes(M::MatMulMat) = axes.(M.factors,(1,2))

function getindex(M::MatMulMat, k::Integer, j::Integer)
    A,B = M.factors
    ret = zero(eltype(M))
    for ℓ in axes(A,2)
        ret += A[k,ℓ] * B[ℓ,j]
    end
    ret
end

getindex(M::MatMulMat, kj::CartesianIndex{2}) = M[kj[1], kj[2]]
