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

        length(M::Applied{<:Any, typeof($op)}) = prod(size(M))
        applied_size(::typeof($op), args...) = size(first(args))
        applied_axes(::typeof($op), args...) = axes(first(args))
    end
end


getindex(M::Add, k::Integer) = sum(getindex.(M.args, k))
getindex(M::Add, k::Integer, j::Integer) = sum(getindex.(M.args, k, j))

getindex(M::Add, k::CartesianIndex{1}) = M[convert(Int, k)]
getindex(M::Add, kj::CartesianIndex{2}) = M[kj[1], kj[2]]

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
            for a in arguments(A)
                C .= applied(+,applied(*,α, a,B), C)
            end
            C
        end
        function materialize!(M::$MulAdd_{ApplyLayout{typeof(-)}})
            α, A, B, β, C = M.α, M.A, M.B, M.β, M.C
            if C ≡ B
                B = copy(B)
            end
            _fill_lmul!(β, C)
            a1,a2 = arguments(A)
            C .= applied(+,applied(*,α, a1,B), C)
            C .= applied(+,applied(*,-α, a2,B), C)
            C
        end
    end
end


###
# views
####
_view_tuple(a, b::Tuple) = view(a, b...)
for op in (:+, :-)
    @eval begin
        sublayout(a::ApplyLayout{typeof($op)}, _) = a
        arguments(::ApplyLayout{typeof($op)}, a::SubArray) =
            _view_tuple.(arguments(parent(a)), Ref(parentindices(a)))
        call(::ApplyLayout{typeof($op)}, a::SubArray) = $op
    end
end


###
# support BroadcastLayout
###

_broadcasted_mul(::Tuple{}, _) = ()
_broadcasted_mul(_, ::Tuple{}) = ()
_broadcasted_mul(a::Tuple{Number,Vararg{Any}}, b::AbstractVector) = (first(a)*sum(b), _broadcasted_mul(tail(a), b)...)
_broadcasted_mul(a::Tuple{Number,Vararg{Any}}, B::AbstractMatrix) = (first(a)*sum(B; dims=1), _broadcasted_mul(tail(a), B)...)
_broadcasted_mul(a::Tuple{AbstractVector,Vararg{Any}}, b::AbstractVector) = (first(a)*sum(b), _broadcasted_mul(tail(a), b)...)
_broadcasted_mul(a::Tuple{AbstractVector,Vararg{Any}}, B::AbstractMatrix) = (first(a)*sum(B; dims=1), _broadcasted_mul(tail(a), B)...)
_broadcasted_mul(A::Tuple{AbstractMatrix,Vararg{Any}}, b::AbstractVector) = (size(first(A),2) == 1 ? vec(first(A))*sum(b) : (first(A)*b), _broadcasted_mul(tail(A), b)...)
_broadcasted_mul(A::Tuple{AbstractMatrix,Vararg{Any}}, B::AbstractMatrix) = (size(first(A),2) == 1 ? first(A)*sum(B; dims=1) : (first(A)*B), _broadcasted_mul(tail(A), B)...)
_broadcasted_mul(A::AbstractMatrix, b::Tuple{Number,Vararg{Any}}) = (sum(A; dims=2)*first(b)[1], _broadcasted_mul(A, tail(b))...)
_broadcasted_mul(A::AbstractMatrix, b::Tuple{AbstractVector,Vararg{Any}}) = (size(first(b),1) == 1 ? (sum(A; dims=2)*first(b)[1]) : (A*first(b)), _broadcasted_mul(A, tail(b))...)
_broadcasted_mul(A::AbstractMatrix, B::Tuple{AbstractMatrix,Vararg{Any}}) = (size(first(B),1) == 1 ? (sum(A; dims=2) * first(B)) : (A * first(B)), _broadcasted_mul(A, tail(B))...)


for op in (:+, :-)
    @eval begin
        simplifiable(M::Mul{BroadcastLayout{typeof($op)}}) = Val(true)
        simplifiable(M::Mul{<:Any,BroadcastLayout{typeof($op)}}) = Val(true)
        simplifiable(M::Mul{BroadcastLayout{typeof($op)},BroadcastLayout{typeof($op)}}) = simplifiable(Mul{BroadcastLayout{typeof($op)},UnknownLayout}(M.A, M.B))
        copy(M::Mul{BroadcastLayout{typeof($op)}}) = broadcast($op, _broadcasted_mul(arguments(BroadcastLayout{typeof($op)}(), M.A), M.B)...)
        copy(M::Mul{BroadcastLayout{typeof($op)},<:LazyLayouts}) = broadcast($op, _broadcasted_mul(arguments(BroadcastLayout{typeof($op)}(), M.A), M.B)...)
        copy(M::Mul{<:Any,BroadcastLayout{typeof($op)}}) = broadcast($op, _broadcasted_mul(M.A, arguments(BroadcastLayout{typeof($op)}(), M.B))...)
        copy(M::Mul{<:LazyLayouts,BroadcastLayout{typeof($op)}}) = broadcast($op, _broadcasted_mul(M.A, arguments(BroadcastLayout{typeof($op)}(), M.B))...)
        copy(M::Mul{BroadcastLayout{typeof($op)},BroadcastLayout{typeof($op)}}) = copy(Mul{BroadcastLayout{typeof($op)},UnknownLayout}(M.A, M.B))
        copy(M::Mul{BroadcastLayout{typeof($op)},ApplyLayout{typeof(*)}}) = copy(Mul{BroadcastLayout{typeof($op)},UnknownLayout}(M.A, M.B))
        copy(M::Mul{ApplyLayout{typeof(*)},BroadcastLayout{typeof($op)}}) = copy(Mul{UnknownLayout,BroadcastLayout{typeof($op)}}(M.A, M.B))
        simplify(M::Mul{BroadcastLayout{typeof($op)}}) = copy(Mul{BroadcastLayout{typeof($op)},UnknownLayout}(M.A, M.B)) # TODO: remove, here for back-compat with QuasiArrays.jl
        simplify(M::Mul{<:Any,BroadcastLayout{typeof($op)}}) = copy(Mul{UnknownLayout,BroadcastLayout{typeof($op)}}(M.A, M.B)) # TODO: remove, here for back-compat with QuasiArrays.jl
        simplify(M::Mul{BroadcastLayout{typeof($op)},BroadcastLayout{typeof($op)}}) = copy(M) # TODO: remove, here for back-compat with QuasiArrays.jl
    end
end

simplifiable(M::Mul{BroadcastLayout{typeof(+)},BroadcastLayout{typeof(-)}}) = simplifiable(Mul{BroadcastLayout{typeof(+)},UnknownLayout}(M.A, M.B))
simplifiable(M::Mul{BroadcastLayout{typeof(-)},BroadcastLayout{typeof(+)}}) = simplifiable(Mul{BroadcastLayout{typeof(-)},UnknownLayout}(M.A, M.B))

copy(M::Mul{BroadcastLayout{typeof(+)},BroadcastLayout{typeof(-)}}) = copy(Mul{BroadcastLayout{typeof(+)},UnknownLayout}(M.A, M.B))
copy(M::Mul{BroadcastLayout{typeof(-)} ,BroadcastLayout{typeof(+)}}) = copy(Mul{BroadcastLayout{typeof(-)} ,UnknownLayout}(M.A, M.B))
simplify(M::Mul{BroadcastLayout{typeof(+)},BroadcastLayout{typeof(-)}}) = copy(Mul{BroadcastLayout{typeof(+)},UnknownLayout}(M.A, M.B)) # TODO: remove, here for back-compat with QuasiArrays.jl
simplify(M::Mul{BroadcastLayout{typeof(-)} ,BroadcastLayout{typeof(+)}}) = copy(Mul{BroadcastLayout{typeof(-)} ,UnknownLayout}(M.A, M.B)) # TODO: remove, here for back-compat with QuasiArrays.jl
