struct Mul{Styles<:Tuple, Factors<:Tuple}
    styles::Styles
    factors::Factors
end


"""
 Mul(A1, A2, …, AN)

represents lazy multiplication A1*A2*…*AN. The factors must have compatible axes.
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

similar(M::Mul) = similar(M, eltype(M))


# default is to stay Lazy
materialize(M::Mul2) = M


_materialize2(A...) = _materialize(A...)
_materialize2(A::Mul, B...) = A * _materialize(B...)
_materialize(A) = materialize(A)
_materialize(A, B) = materialize(Mul(A,B))
_materialize(A, B, C, D...) = _materialize2(materialize(Mul(A,B)), C, D...)
materialize(M::Mul) = _materialize(M.factors...)

*(A::Mul, B::Mul) = Mul(A.factors..., B.factors...)
*(A::Mul, B) = Mul(A.factors..., B)
*(A, B::Mul) = Mul(A, B.factors...)


####
# Matrix * Array
####

const ArrayMulArray{styleA, styleB, p, q, T, V} =
    Mul2{styleA, styleB, <:AbstractArray{T,p}, <:AbstractArray{V,q}}

# the default is always Array
similar(M::ArrayMulArray, ::Type{T}) where T = Array{T}(undef, size(M))
materialize(M::ArrayMulArray) = copyto!(similar(M), M)

@inline copyto!(dest::AbstractArray, M::Mul) = _copyto!(MemoryLayout(dest), dest, M)


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
    @inbounds for ℓ in axes(A,2)
        ret += A[k,ℓ] * B[ℓ,j]
    end
    ret
end

getindex(M::MatMulMat, kj::CartesianIndex{2}) = M[kj[1], kj[2]]
