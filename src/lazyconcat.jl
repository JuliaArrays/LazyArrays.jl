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


## copyto!
# based on Base/array.jl, Base/abstractarray.jl

function copyto!(dest::AbstractMatrix, V::Vcat{<:Any,2})
    arrays = V.arrays
    nargs = length(arrays)
    nrows = size(dest,1)
    nrows == sum(a->size(a, 1), arrays) || throw(DimensionMismatch("sum of rows each matrix must equal $nrows"))
    ncols = size(dest, 2)
    for a in arrays
        if size(a, 2) != ncols
            throw(DimensionMismatch("number of columns of each array must match (got $(map(x->size(x,2), A)))"))
        end
    end
    pos = 1
    for a in arrays
        p1 = pos+size(a,1)-1
        dest[pos:p1, :] = a
        pos = p1+1
    end
    return dest
end

function copyto!(arr::AbstractVector, A::Vcat{<:Any,1,<:Tuple{Vararg{<:AbstractVector}}})
    arrays = A.arrays
    n = 0
    for a in arrays
        n += length(a)
    end
    n == length(arr) || throw(DimensionMismatch("destination must have length equal to sums of concatenated vectors"))

    i = 0
    @inbounds for a in arrays
        for ai in a
            i += 1
            arr[i] = ai
        end
    end
    arr
end

function copyto!(arr::Vector{T}, A::Vcat{T,1,<:Tuple{Vararg{<:Vector{T}}}}) where T
    arrays = A.arrays
    n = 0
    for a in arrays
        n += length(a)
    end
    n == length(arr) || throw(DimensionMismatch("destination must have length equal to sums of concatenated vectors"))

    ptr = pointer(arr)
    if isbitstype(T)
        elsz = Core.sizeof(T)
    elseif isbitsunion(T)
        elsz = bitsunionsize(T)
        selptr = convert(Ptr{UInt8}, ptr) + n * elsz
    else
        elsz = Core.sizeof(Ptr{Cvoid})
    end
    t = @_gc_preserve_begin arr
    for a in arrays
        na = length(a)
        nba = na * elsz
        if isbitstype(T)
            ccall(:memcpy, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, UInt),
                  ptr, a, nba)
        elseif isbitsunion(T)
            ccall(:memcpy, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, UInt),
                  ptr, a, nba)
            # copy selector bytes
            ccall(:memcpy, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, UInt),
                  selptr, convert(Ptr{UInt8}, pointer(a)) + nba, na)
            selptr += na
        else
            ccall(:jl_array_ptr_copy, Cvoid, (Any, Ptr{Cvoid}, Any, Ptr{Cvoid}, Int),
                  arr, ptr, a, pointer(a), na)
        end
        ptr += nba
    end
    @_gc_preserve_end t
    return arr
end

function copyto!(dest::AbstractMatrix, H::Hcat)
    arrays = H.arrays
    nargs = length(arrays)
    nrows = size(dest, 1)
    ncols = 0
    dense = true
    for a in arrays
        dense &= isa(a,Array)
        nd = ndims(a)
        ncols += (nd==2 ? size(a,2) : 1)
    end

    nrows == size(H,1) || throw(DimensionMismatch("Destination rows must match"))
    ncols == size(dest,2) || throw(DimensionMismatch("Destination columns must match"))

    pos = 1
    if dense
        for a in arrays
            n = length(a)
            copyto!(dest, pos, a, 1, n)
            pos += n
        end
    else
        for a in arrays
            p1 = pos+(isa(a,AbstractMatrix) ? size(a, 2) : 1)-1
            dest[:, pos:p1] = a
            pos = p1+1
        end
    end
    return dest
end

function copyto!(dest::AbstractMatrix, H::Hcat{<:Any,Tuple{Vararg{<:AbstractVector}}})
    height = size(dest, 1)
    for j = 1:length(H)
        if length(H[j]) != height
            throw(DimensionMismatch("vectors must have same lengths"))
        end
    end
    for j=1:length(H)
        dest[i,:] .= H[j]
    end

    dest
end
