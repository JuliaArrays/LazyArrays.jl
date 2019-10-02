####
# These are special routines to make operations involving +
# more efficient
####


const Add{Factors<:Tuple} = Applied{<:Any, typeof(+), Factors}

const AddArray{T,N,Factors<:Tuple} = ApplyArray{T,N,typeof(+), Factors}
const AddVector{T,Factors<:Tuple} = AddArray{T,1,Factors}
const AddMatrix{T,Factors<:Tuple} = AddArray{T,2,Factors}

AddArray(factors...) = ApplyArray(+, factors...)

"""
    Add(A1, A2, …, AN)

A lazy representation of `A1 + A2 + … + AN`; i.e., a shorthand for `applied(+, A1, A2, …, AN)`.
"""
Add(As...) = applied(+, As...)



for op in (:+, :-)
    @eval begin
        size(M::Applied{<:Any, typeof($op)}, p::Int) = size(M)[p]
        axes(M::Applied{<:Any, typeof($op)}, p::Int) = axes(M)[p]
        ndims(M::Applied{<:Any, typeof($op)}) = ndims(first(M.args))

        length(M::Applied{<:Any, typeof($op)}) = prod(size(M))
        size(M::Applied{<:Any, typeof($op)}) = length.(axes(M))
        axes(M::Applied{<:Any, typeof($op)}) = axes(first(M.args))

        eltype(M::Applied{<:Any, typeof($op)}) = promote_type(map(eltype,M.args)...)

        combine_mul_styles(::ApplyLayout{typeof($op)}) = IdentityMulStyle()
    end
end


getindex(M::Add, k::Integer) = sum(getindex.(M.args, k))
getindex(M::Add, k::Integer, j::Integer) = sum(getindex.(M.args, k, j))

getindex(M::Add, k::CartesianIndex{1}) = M[convert(Int, k)]
getindex(M::Add, kj::CartesianIndex{2}) = M[kj[1], kj[2]]

zero!(A::AbstractArray{T}) where T = fill!(A,zero(T))
function zero!(A::AbstractArray{<:AbstractArray}) 
    for a in A
        zero!(a)
    end
    A
end

_fill_lmul!(β, A::AbstractArray{T}) where T = iszero(β) ? zero!(A) : lmul!(β, A)
for MulAdd_ in [MatMulMatAdd, MatMulVecAdd]
    # `MulAdd{ApplyLayout{typeof(+)}}` cannot "win" against
    # `MatMulMatAdd` and `MatMulVecAdd` hence `@eval`:
    @eval begin
        function materialize!(M::$MulAdd_{ApplyLayout{typeof(+)}})
            α, A, B, β, C = M.α, M.A, M.B, M.β, M.C
            if C ≡ B
                B = copy(B)
            end
            _fill_lmul!(β, C)
            for A in A.args
                C .= applied(+,applied(*,α, A,B), C)
            end
            C
        end
        function materialize!(M::$MulAdd_{ApplyLayout{typeof(-)}})
            α, A, B, β, C = M.α, M.A, M.B, M.β, M.C
            if C ≡ B
                B = copy(B)
            end
            _fill_lmul!(β, C)
            a1,a2 = A.args
            C .= applied(+,applied(*,α, a1,B), C)
            C .= applied(-,applied(*,α, a2,B), C)
            C
        end
    end
end
