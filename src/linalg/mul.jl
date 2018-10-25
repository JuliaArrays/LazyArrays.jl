struct Mul{StyleA, StyleB, AType, BType}
    style_A::StyleA
    style_B::StyleB
    A::AType
    B::BType
end

Mul(A, x) = Mul(MemoryLayout(A), MemoryLayout(x), A, x)

eltype(M::Mul{<:Any,<:Any,AT,BT}) where {AT,BT} = Base.promote_op(*, eltype(AT), eltype(BT))

####
# Matrix * Array
####

const ArrayMulArray{styleA, styleB, p, q, T, V} =
    Mul{styleA, styleB, <:AbstractArray{T,p}, <:AbstractArray{V,q}}

@inline copyto!(dest::AbstractArray, M::Mul) = _copyto!(MemoryLayout(dest), dest, M)
# default to Base mul!
function _copyto!(_, dest::AbstractArray, M::ArrayMulArray)
    A,x = M.A, M.B
    mul!(dest, A, x)
end


####
# Matrix * Vector
####
const MatMulVec{styleA, styleB, T, V} = ArrayMulArray{styleA, styleB, 2, 1, T, V}

length(M::MatMulVec) = size(M.A,1)
axes(M::MatMulVec) = (axes(M.A,1),)

function getindex(M::MatMulVec, k::Integer)
    ret = zero(eltype(M))
    for j = 1:size(M.A,2)
        ret += M.A[k,j] * M.B[j]
    end
    ret
end

getindex(M::MatMulVec, k::CartesianIndex{1}) = M[convert(Int, k)]



####
# Matrix * Matrix
####

MatMulMat{styleA, styleB, T, V} = ArrayMulArray{styleA, styleB, 2, 2, T, V}



size(M::MatMulMat) = (size(M.A,1),size(M.B,2))
axes(M::MatMulMat) = (axes(M.A,1),axes(M.B,2))

function getindex(M::MatMulMat, k::Integer, j::Integer)
    ret = zero(eltype(M))
    for ℓ in axes(M.A,2)
        ret += M.A[k,ℓ] * M.B[ℓ,j]
    end
    ret
end

getindex(M::MatMulMat, kj::CartesianIndex{2}) = M[kj[1], kj[2]]
