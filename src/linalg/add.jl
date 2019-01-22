####
# These are special routines to make operations involving +
# more efficient
####


const Add{Factors<:Tuple} = Applied{<:Any, typeof(+), Factors}

size(M::Add, p::Int) = size(M)[p]
axes(M::Add, p::Int) = axes(M)[p]
ndims(M::Add) = ndims(first(M.args))

length(M::Add) = prod(size(M))
size(M::Add) = length.(axes(M))
axes(M::Add) = axes(first(M.args))


eltype(M::Add) = Base._return_type(+, eltype.(M.args))

const AddArray{T,N,Factors<:Tuple} = ApplyArray{T,N,<:Add{Factors}}
const AddVector{T,Factors<:Tuple} = AddArray{T,1,Factors}
const AddMatrix{T,Factors<:Tuple} = AddArray{T,2,Factors}

AddArray(factors...) = ApplyArray(+, factors...)

"""
    Add(A1, A2, …, AN)

A lazy representation of `A1 + A2 + … + AN`; i.e., a shorthand for `applied(+, A1, A2, …, AN)`.
"""
Add(As...) = applied(+, As...)


for MulAdd_ in [MatMulMatAdd, MatMulVecAdd]
    # `MulAdd{<:ApplyLayout{typeof(+)}}` cannot "win" against
    # `MatMulMatAdd` and `MatMulVecAdd` hence `@eval`:
    @eval function materialize!(M::$MulAdd_{<:ApplyLayout{typeof(+)}})
        α, A, B, β, C = M.α, M.A, M.B, M.β, M.C
        if C ≡ B
            B = copy(B)
        end
        lmul!(β, C)
        for A in A.applied.args
            C .= α .* Mul(A, B) .+ C
        end
        C
    end
end
