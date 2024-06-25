
const Kron{T,N,I<:Tuple} = ApplyArray{T,N,typeof(kron),I}

function instantiate(K::Applied{<:Any,typeof(kron)})
    isempty(K.args) && throw(ArgumentError("Cannot take kronecker product of empty vectors"))
    applied(kron, map(instantiate,K.args)...)
end

Kron(A...) = ApplyArray(kron, A...)
Kron{T}(A...) where T = ApplyArray{T}(kron, A...)

_kron_dims() = 0
_kron_dims(A, B...) = max(ndims(A), _kron_dims(B...))

applied_eltype(::typeof(kron), args...) = promote_type(map(eltype,args)...)
applied_ndims(::typeof(kron), args...) = _kron_dims(args...)

size(K::Kron, j::Int) = prod(size.(K.args, j))
size(a::Kron{<:Any,1}) = (size(a,1),)
size(a::Kron{<:Any,2}) = (size(a,1), size(a,2))
axes(a::Kron{<:Any,1}) = (oneto(size(a,1)),)
axes(a::Kron{<:Any,2}) = (oneto(size(a,1)), oneto(size(a,2)))


issquare(M::AbstractMatrix) = (size(M, 1) == size(M, 2))

adjoint(K::Kron{<:Any, 2}) = Kron(adjoint.(K.args)...)
transpose(K::Kron{<:Any, 2}) = Kron(transpose.(K.args)...)
pinv(K::Kron{<:Number, 2}) = Kron(pinv.(K.args)...)
function inv(K::Kron{<:Number, 2})
    n = checksquare(K)

    # see det below for why the presence of rectangular factors renders
    # the kronecker product singular
    all(issquare, K.args) || throw(LinearAlgebra.SingularException(n))

    return Kron(inv.(K.args)...)
end


function det(K::Kron{T, 2}) where T<:Number
    s = checksquare(K)
    d = one(T)

    for A in K.args
        if issquare(A)
            dA = det(A)
            if iszero(dA)
                return dA
            end
            d *= dA^(s ÷ size(A, 1))
        else
            # The Kronecker Product of rectangular matrices, if it is square, will
            # have determinant zero. This can be shown by using the fact that
            # rank(A ⊗ B) = rank(A)rank(B) and showing that this is strictly less
            # than the number of rows/cols in the resulting Kronecker matrix. Hence,
            # since A ⊗ B does not have full rank, its determinant must be zero.
            return zero(T)
        end
    end
    return d
end


function logabsdet(K::Kron{T, 2}) where T<:Number
    s = checksquare(K)
    d = zero(T)
    sgn = one(T)

    for A in K.args
        if issquare(A)
            ldA, sgnA = logabsdet(A)
            if isinf(ldA)
                return ldA, sgnA
            end
            p = (s ÷ size(A, 1))
            d += p * ldA
            sgn *= sgnA ^ p
        else
            # see definition of det above for explanation
            return float(T)(-Inf), zero(T)
        end
    end
    return d, sgn
end


function tr(K::Kron{<:Any, 2})
    checksquare(K)
    if all(issquare, K.args)
        return prod(tr.(K.args))
    else
        return sum(diag(K))
    end
end


function diag(K::Kron{<:Any, 2})
    if all(issquare, K.args)
        return kron(diag.(K.args)...)
    else
        d = similar(K.args[1], minimum(size(K)))
        @inbounds for i in 1:length(d)
            d[i] = K[i, i]
        end
        return d
    end
end


_compatible_sizes((A, B)) = (size(A, 2) == size(B, 1))


# Implements the mixed-product property for the Kronecker product and matrix
# multiplication
function copy(M::Mul{ApplyLayout{typeof(kron)}, ApplyLayout{typeof(kron)}})
    A, B = M.A, M.B
    MT = promote_type(map(typeof, arguments(A))..., map(typeof, arguments(B))...)
    # Keeping it simple for now, but could potentially make this "alignment"-check
    # more precise. For example, if one factor of A has 6 columns and it's aligned
    # with two factors of B with 2 and 3 rows, then we can still use the
    # mixed-product property after explicitly computing the Kronecker product of
    # the two factors of B (or maybe just deferring to the shuffle algorithm)
    if (length(arguments(A)) == length(arguments(B))) && all(_compatible_sizes, zip(arguments(A), arguments(B)))
        factors = [(A_i * B_i) for (A_i, B_i) in zip(arguments(A), arguments(B))]
        return Kron(factors...)
    else
        algo_type = shuffle_algorithm_type(MT)
        return shuffle_algorithm(algo_type, A, B, eltype(M))
    end
end


# Implements Roth's lemma aka the "vec-trick" for multiplying a 2-factor Kron
# matrix to a vector
function copy(M::Mul{
    LazyArrays.ApplyLayout{typeof(kron)},
    <:Any,
    Kron{T,2,NTuple{2,M}},V
} where {T, M<:AbstractMatrix{T}, V<:AbstractVector})
    K, v = M.A, M.B
    A, B = arguments(K)
    V = reshape(v, size(B, 2), size(A, 2))
    return vec(B * V * transpose(A))
end



abstract type ShuffleAlgorithm end
struct Shuffle <: ShuffleAlgorithm end
struct ModifiedShuffle <: ShuffleAlgorithm end

shuffle_algorithm_type(::Type) = Shuffle()
shuffle_algorithm_type(::Type{<:AbstractArray}) = ModifiedShuffle()
shuffle_algorithm_type(::Type{<:ApplyArray}) = Shuffle()


function copy(M::Mul{ApplyLayout{typeof(kron)}})
    MT = promote_type(map(typeof, arguments(M.A))...)
    algo_type = shuffle_algorithm_type(MT)
    return shuffle_algorithm(algo_type, M.A, M.B, eltype(M))
end


function shuffle_algorithm(
    ::ModifiedShuffle, K::Kron{T,2} where T, p::AbstractVecOrMat, OT::Type{<:Number}
)

    if size(K, 2) != size(p, 1)
        if ndims(p) > 1
            throw(DimensionMismatch("matrix A has dimensions $(size(K)), matrix B has dimensions $(size(p))"))
        else
            throw(DimensionMismatch("matrix A has dimensions $(size(K)), vector B has length $(size(p, 1))"))
        end
    end

    output_sizes = (ndims(p) > 1) ? (size(K, 1), size(p, 2)) : (size(K,1),)
    q = fill!(similar(p, OT, output_sizes), zero(OT))
    if any(iszero, K.args)
        return q
    end

    K_shrunk_factors::Vector{eltype(K.args)} = []
    R_H::Vector{Vector{Int}} = []
    C_H::Vector{Vector{Int}} = []

    is_dense = !any(issparse, K.args)

    # note: the following computation costs are for multiplication against
    #       a single vector.
    nnz_ = [issparse(X_h) ? nnz(X_h) : prod(size(X_h)) for X_h in K.args]
    trad_cost = 2*sum([
        prod(size.(K.args[1:h-1], 2)) * nnz_[h] * prod(size.(K.args[h+1:end], 1))
        for (h, X_h) in enumerate(K.args)
        if X_h != I
    ])

    naive_cost = 2*prod(size(K))

    if is_dense && naive_cost <= trad_cost && naive_cost < 1e6
        return Matrix(K) * p
    end

    for X_h in K.args
        R_h, C_h = Set{Int}(), Set{Int}()

        for i in axes(X_h, 1), j in axes(X_h, 2)
            if X_h[i, j] != 0
                push!(R_h, i)
                push!(C_h, j)
            end
        end

        R_h, C_h = sort(collect(R_h)), sort(collect(C_h))

        is_dense = is_dense && (axes(X_h, 1) == R_h && axes(X_h, 2) == C_h)

        push!(K_shrunk_factors, X_h[R_h, C_h])
        push!(R_H, R_h)
        push!(C_H, C_h)
    end

    if is_dense
        # this means that the component matrices were all dense AND they had no
        # zero rows/cols
        return shuffle_algorithm(Shuffle(), K, p, OT)
    end

    nnz_m = [issparse(X_h) ? nnz(X_h) : prod(size(X_h)) for X_h in K_shrunk_factors]
    modified_cost = 2*sum([
        prod(length.(C_H[1:h-1])) * nnz_m[h] * prod(length.(R_H[h+1:end]))
        for (h, X_h) in enumerate(K.args)
        if X_h != I
    ])

    if trad_cost <= modified_cost
        return shuffle_algorithm(Shuffle(), K, p, OT)
    end

    R_indices = Iterators.product(reverse(R_H)...)
    C_indices = Iterators.product(reverse(C_H)...)

    r_index_v = [1, cumprod([size(m, 1) for m in K.args[end:-1:2]])...]
    c_index_v = [1, cumprod([size(m, 2) for m in K.args[end:-1:2]])...]

    r_indices = zeros(Int, prod(size(R_indices)))
    for (i, r_ind) in enumerate(R_indices)
        r_indices[i] = dot(r_index_v, r_ind .- 1) + 1
    end

    c_indices = zeros(Int, prod(size(C_indices)))
    for (i, c_ind) in enumerate(C_indices)
        c_indices[i] = dot(c_index_v, c_ind .- 1) + 1
    end

    K_ = Kron(K_shrunk_factors...)
    if ndims(p) == 1
        q[r_indices] = shuffle_algorithm(Shuffle(), K_, p[c_indices], OT)
    else
        q[r_indices, :] = shuffle_algorithm(Shuffle(), K_, p[c_indices, :], OT)
    end

    return q
end


function shuffle_algorithm(::Shuffle, K::Kron{T,2} where T, p::AbstractVecOrMat, OT::Type{<:Number})
    q = copy!(similar(p, OT), p)
    r = [size(m, 1) for m in K.args]
    c = [size(m, 2) for m in K.args]

    all_square = all(map(==, r, c))
    is_vector = (ndims(p) == 1)

    i_left, i_right = 1, prod(c)
    H = length(K.args)

    @inbounds for h in 1:H
        X_h = K.args[h]
        r_h, c_h = r[h], c[h]
        i_right ÷= c_h

        if X_h != I
            if all_square
                q′ = q
            else
                size_ = prod(r[1:h]) * prod(c[h+1:H])
                output_sizes = (ndims(p) > 1) ? (size_, size(p, 2)) : (size_,)
                q′ = fill!(similar(p, OT, output_sizes), zero(OT))
            end

            base_i, base_j = 0, 0
            for i_l in 1:i_left
                for i_r in 1:i_right
                    slc_in  = base_i + i_r : i_right : base_i + i_r + (c_h-1)*i_right
                    slc_out = base_j + i_r : i_right : base_j + i_r + (r_h-1)*i_right

                    if is_vector
                        @views q′[slc_out] = X_h * q[slc_in]
                    else
                        @views q′[slc_out, :] = X_h * q[slc_in, :]
                    end
                end

                base_i += i_right*c_h
                base_j += i_right*r_h
            end
            q = q′
        end

        i_left *= r_h
    end

    return q
end



kron_getindex((A,)::Tuple{AbstractVector}, k::Integer) = A[k]
function kron_getindex((A,B)::NTuple{2,AbstractVector}, k::Integer)
    K,κ = divrem(k-1, length(B))
    A[K+1]*B[κ+1]
end
kron_getindex((A,)::Tuple{AbstractMatrix}, k::Integer, j::Integer) = A[k,j]
function kron_getindex((A,B)::NTuple{2,AbstractArray}, k::Integer, j::Integer)
    K,κ = divrem(k-1, size(B,1))
    J,ξ = divrem(j-1, size(B,2))
    A[K+1,J+1]*B[κ+1,ξ+1]
end

kron_getindex(args::Tuple, k::Integer, j::Integer) = kron_getindex(tuple(Kron(args[1:2]...), args[3:end]...), k, j)
kron_getindex(args::Tuple, k::Integer) = kron_getindex(tuple(Kron(args[1:2]...), args[3:end]...), k)

getindex(K::Kron{<:Any,1}, k::Integer) = kron_getindex(K.args, k)
getindex(K::Kron{<:Any,2}, k::Integer, j::Integer) = kron_getindex(K.args, k, j)
getindex(K::Applied{DefaultArrayApplyStyle,typeof(kron)}, k::Integer) = kron_getindex(K.args, k)
getindex(K::Applied{DefaultArrayApplyStyle,typeof(kron)}, k::Integer, j::Integer) = kron_getindex(K.args, k, j)




## Adapted from julia/stdlib/LinearAlgebra/src/dense.jl kron definition
function _kron!(R, a, b)
    size(R) == size(Kron(a,b)) || throw(DimensionMismatch("Matrices have wrong dimensions"))
    require_one_based_indexing(a, b)
    m = 1
    @inbounds for j = 1:size(a,2), l = 1:size(b,2), i = 1:size(a,1)
        aij = a[i,j]
        for k = 1:size(b,1)
            R[m] = aij*b[k,l]
            m += 1
        end
    end
    R
end

_kron!(R, a) = copyto!(R, a)

_kron!(R, a, b, c, d...) = copyto!_layout(UnknownLayout(), UnknownLayout(), R, Kron(a,b,c,d...))

copyto!_layout(_, ::ApplyLayout{typeof(kron)}, R::AbstractMatrix, K::AbstractMatrix) =
    _kron!(R, arguments(K)...)
copyto!_layout(_, ::ApplyLayout{typeof(kron)}, R::AbstractVector, K::AbstractVector) =
    _kron!(R, arguments(K)...)


struct Diff{T, N, Arr} <: LazyArray{T, N}
    v::Arr
    dims::Int
end

Diff(v::AbstractVector{T}) where T = Diff{T,1,typeof(v)}(v, 1)

function Diff(A::AbstractMatrix{T}; dims::Integer) where T
    dims == 1 || dims == 2 || throw(ArgumentError("dimension must be 1 or 2, got $dims"))
    Diff{T,2,typeof(A)}(A, dims)
end

IndexStyle(::Type{<:Diff{<:Any,1}}) = IndexLinear()

size(D::Diff{<:Any,1}) = (length(D.v)-1,)
function size(D::Diff{<:Any,2})
    if D.dims == 1
        (size(D.v,1)-1,size(D.v,2))
    else #dims == 2
        (size(D.v,1),size(D.v,2)-1)
    end
end

getindex(D::Diff{<:Any, 1}, k::Integer) = D.v[k+1] - D.v[k]
function getindex(D::Diff, k::Integer, j::Integer)
    if D.dims == 1
        D.v[k+1,j] - D.v[k,j]
    else # dims == 2
        D.v[k,j+1] - D.v[k,j]
    end
end

mutable struct Accumulate{T, N, Op, DM<:AbstractArray{T,N}, Arr<:AbstractArray{T,N}} <: AbstractCachedArray{T, N}
    op::Op
    data::DM
    v::Arr
    dims::Int
    datasize::NTuple{N,Int}
end

Accumulate(op, data::AbstractArray, v::AbstractArray, d::Int) = Accumulate(op, data, v, d, size(data))
function Accumulate(op, v::AbstractVector, d::Int)
    @assert d == 1
    Accumulate(op, [v[1]], v, d)
end

function Accumulate(op, v::AbstractMatrix, d::Int)
    @assert 1 ≤ d ≤ 2
    data = d == 1 ? v[1:1,:] : v[:,1:1]
    Accumulate(op, data, v, d)
end
Accumulate(op, v::AbstractVector{T}; dims::Integer=1) where T = Accumulate(op, v, dims)

function Accumulate(op, A::AbstractMatrix{T}; dims::Integer=1) where T
    dims == 1 || dims == 2 || throw(ArgumentError("dimension must be 1 or 2, got $dims"))
    Accumulate(op, A, dims)
end

const Cumsum{T,N,Arr} = Accumulate{T,N,typeof(+),Array{T,N},Arr}
Cumsum(v::AbstractArray; dims::Integer=1) = Accumulate(+, v; dims=dims)

IndexStyle(::Type{<:Accumulate{<:Any,1}}) = IndexLinear()

size(Q::Accumulate) = size(Q.v)
axes(Q::Accumulate) = axes(Q.v)

copy(Q::Accumulate) = Accumulate(Q.op, copy(Q.data), copy(Q.v), Q.dims, Q.datasize)


==(a::Cumsum{<:Any,1}, b::Cumsum{<:Any,1}) = a.v == b.v

copyto!(x::AbstractArray{<:Any,N}, C::Cumsum{<:Any,N}) where N = cumsum!(x, C.v)

# How we populate the data
function cache_filldata!(K::Accumulate{<:Any,1}, inds)
    copyto!(view(K.data,inds), view(K.v,inds))
    @inbounds for k in inds
        K.data[k] = K.op(K.data[k-1], K.data[k])
    end
end

function cache_filldata!(K::Accumulate{<:Any,2}, kr, jr)
    if K.dims == 1
        @inbounds for j in jr, k in kr
            K.data[k,j] = K.op(K.data[k-1,j], K.v[k,j])
        end
    else # K.dims == 2
        @inbounds for j in jr, k in kr
            K.data[k,j] = K.op(K.data[k,j-1], K.v[k,j])
        end
    end
end


# keep lazy
cumsum(a::LazyArray; kwds...) = Cumsum(a; kwds...)
accumulate(op, a::LazyArray; kwds...) = Accumulate(op, a; kwds...)


const AccumulateAbstractVector{T,Op} = Accumulate{T,1,Op,Vector{T},AbstractVector{T}}
AccumulateAbstractVector(op, A::AbstractVector{T}) where T = AccumulateAbstractVector{T,typeof(op)}(op, [A[1]], A, 1, (1,))


## Rotations

for op in (:rot180, :rotl90, :rotr90)
    @eval begin
        applied_ndims(::typeof($op), a) = 2
        applied_eltype(::typeof($op), a) = eltype(a)
    end
end
applied_size(::typeof(rot180), a) = size(a)
applied_axes(::typeof(rot180), a) = axes(a)
for op in (:rotl90, :rotr90)
    @eval begin
        applied_size(::typeof($op), a) = reverse(size(a))
        applied_axes(::typeof($op), a) = reverse(axes(a))
    end
end

getindex(A::Applied{<:Any,typeof(rot180)}, k::Int, j::Int) = A.args[1][end-k+1,end-j+1]
getindex(A::Applied{<:Any,typeof(rotl90)}, k::Int, j::Int) = A.args[1][j,end-k+1]
getindex(A::Applied{<:Any,typeof(rotr90)}, k::Int, j::Int) = A.args[1][end-j+1,k]

applylayout(::Type{typeof(rot180)}, ::AbstractStridedLayout) = StridedLayout()
function strides(A::ApplyMatrix{<:Any,typeof(rot180)})
    a,b = strides(A.args...)
    -a,-b
end
unsafe_convert(::Type{Ptr{T}}, A::ApplyMatrix{T,typeof(rot180)}) where T =
    pointer(A.args..., length(A))

applylayout(::Type{typeof(rot180)}, ::ApplyLayout{typeof(*)}) = ApplyLayout{typeof(*)}()
call(::ApplyLayout{typeof(*)}, A::ApplyMatrix{<:Any,typeof(rot180)}) = *
arguments(::ApplyLayout{typeof(*)}, A::ApplyMatrix{<:Any,typeof(rot180)}) = ApplyMatrix.(rot180, arguments(A.args...))
