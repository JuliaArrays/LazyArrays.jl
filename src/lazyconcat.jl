# Lazy concatenation of AbstractVector's.
# Similar to Iterators.Flatten and some code has been reused from julia/base/iterators.jl

function _Vcat end

struct Vcat{T,N,I} <: AbstractArray{T,N}
    arrays::I

    global function _Vcat(::Type{T}, A::I) where {I<:Tuple,T}
        isempty(A) && throw(ArgumentError("Cannot concatenate empty vectors"))
        new{T,1,I}(A)
    end
    global function _Vcat(::Type{T}, A::I) where I<:Tuple{Vararg{<:AbstractMatrix}} where T
        isempty(A) && throw(ArgumentError("Cannot concatenate empty vectors"))
        m = size(A[1],2)
        for k=2:length(A)
            size(A[k],2) == m || throw(ArgumentError("number of columns of each array must match (got $(map(x->size(x,2), A)))"))
        end
        new{T,2,I}(A)
    end
end

_Vcat(A) = _Vcat(promote_eltypeof(A...), A)
Vcat(args...) = _Vcat(args)
size(f::Vcat{<:Any,1}) = tuple(+(length.(f.arrays)...))
size(f::Vcat{<:Any,2}) = (+(map(a -> size(a,1), f.arrays)...), size(f.arrays[1],2))
Base.IndexStyle(::Type{<:Vcat{T,1}}) where T = Base.IndexLinear()
Base.IndexStyle(::Type{<:Vcat{T,2}}) where T = Base.IndexCartesian()

function getindex(f::Vcat{T,1}, k::Integer) where T
    for A in f.arrays
        n = length(A)
        k ≤ n && return T(A[k])::T
        k -= n
    end
    throw(BoundsError("attempt to access $length(f) Vcat array."))
end

function getindex(f::Vcat{T,2}, k::Integer, j::Integer) where T
    for A in f.arrays
        n = size(A,1)
        k ≤ n && return T(A[k,j])::T
        k -= n
    end
    throw(BoundsError("attempt to access $length(f) Vcat array."))
end

reverse(f::Vcat{<:Any,1}) = Vcat((reverse(itr) for itr in reverse(f.arrays))...)


function _Hcat end

struct Hcat{T,I} <: AbstractMatrix{T}
    arrays::I

    global function _Hcat(::Type{T}, A::I) where {I<:Tuple,T}
        isempty(A) && throw(ArgumentError("Cannot concatenate empty vectors"))
        m = size(A[1],1)
        for k=2:length(A)
            size(A[k],1) == m || throw(ArgumentError("number of columns of each array must match (got $(map(x->size(x,2), A)))"))
        end
        new{T,I}(A)
    end
end

_Hcat(A) = _Hcat(promote_eltypeof(A...), A)
Hcat(args...) = _Hcat(args)
size(f::Hcat) = (size(f.arrays[1],1), +(map(a -> size(a,2), f.arrays)...))
Base.IndexStyle(::Type{<:Hcat}) where T = Base.IndexCartesian()

function getindex(f::Hcat{T}, k::Integer, j::Integer) where T
    for A in f.arrays
        n = size(A,2)
        j ≤ n && return T(A[k,j])::T
        j -= n
    end
    throw(BoundsError("attempt to access $(size(f)) Hcat array."))
end
