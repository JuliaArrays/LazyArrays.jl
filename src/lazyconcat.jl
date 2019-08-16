# Lazy concatenation of AbstractVector's.
# Similar to Iterators.Flatten and some code has been reused from julia/base/iterators.jl

function _Vcat end
abstract type AbstractConcatArray{T,N} <: AbstractArray{T,N} end
struct Vcat{T,N,I} <: AbstractConcatArray{T,N}
    arrays::I

    _Vcat(::Type{T}, A::Tuple{}) where {T} = new{T,1,Tuple{}}(A)
    global _Vcat(::Type{T}, A::I) where {I<:Tuple,T} = new{T,1,I}(A)
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
Vcat{T}(args...) where T = _Vcat(T, args)
Vcat() = Vcat{Any}()

convert(::Type{AbstractArray{T}}, F::Vcat{T}) where T = F
convert(::Type{AbstractArray{T,N}}, F::Vcat{T,N}) where {T,N} = F
convert(::Type{AbstractArray{T}}, F::Vcat) where {T} = _Vcat(T, F.arrays)
convert(::Type{AbstractArray{T,N}}, F::Vcat{<:Any,N}) where {T,N} = _Vcat(T, F.arrays)

_abstractarray(::Type{T}, x::AbstractArray) where T = AbstractArray{T}(x)
_abstractarray(::Type{T}, x) where T = T(x)
AbstractArray(F::Vcat) = copy(F)
AbstractArray{T}(F::Vcat) where T = _Vcat(_abstractarray.(T,F.arrays))
AbstractArray{T,N}(F::Vcat{<:Any,N}) where {T,N} = _Vcat(_abstractarray.(T,F.arrays))

copy(x::Vcat) = Vcat(copy.(x.arrays)...)

size(f::Vcat{<:Any,1,Tuple{}}) = (0,)
size(f::Vcat{<:Any,1}) = tuple(+(length.(f.arrays)...))
size(f::Vcat{<:Any,2}) = (+(map(a -> size(a,1), f.arrays)...), size(f.arrays[1],2))
Base.IndexStyle(::Type{<:Vcat{T,1}}) where T = Base.IndexLinear()
Base.IndexStyle(::Type{<:Vcat{T,2}}) where T = Base.IndexCartesian()



@propagate_inbounds @inline function getindex(f::Vcat{T,1}, k::Integer) where T
    κ = k
    for A in f.arrays
        n = length(A)
        κ ≤ n && return convert(T,A[κ])::T
        κ -= n
    end
    throw(BoundsError(f, k))
end

@propagate_inbounds @inline function getindex(f::Vcat{T,2}, k::Integer, j::Integer) where T
    κ = k
    for A in f.arrays
        n = size(A,1)
        κ ≤ n && return convert(T,A[κ,j])::T
        κ -= n
    end
    throw(BoundsError(f, (k,j)))
end

@propagate_inbounds @inline function setindex!(f::Vcat{T,1}, v, k::Integer) where T
    κ = k
    for A in f.arrays
        n = length(A)
        κ ≤ n && return setindex!(A, v, κ)
        κ -= n
    end
    throw(BoundsError(f, k))
end

@propagate_inbounds @inline function setindex!(f::Vcat{T,2}, v, k::Integer, j::Integer) where T
    κ = k
    for A in f.arrays
        n = size(A,1)
        κ ≤ n && return setindex!(A, v, κ, j)
        κ -= n
    end
    throw(BoundsError(f, (k,j)))
end

reverse(f::Vcat{<:Any,1}) = Vcat((reverse(itr) for itr in reverse(f.arrays))...)


function _Hcat end

struct Hcat{T,I} <: AbstractConcatArray{T,2}
    arrays::I

    global function _Hcat(::Type{T}, A::I) where {I<:Tuple,T}
        isempty(A) && throw(ArgumentError("Cannot concatenate empty vectors"))
        m = size(A[1],1)
        for k=2:length(A)
            size(A[k],1) == m || throw(ArgumentError("number of rows of each array must match (got $(map(x->size(x,1), A)))"))
        end
        new{T,I}(A)
    end
end

_Hcat(A) = _Hcat(promote_eltypeof(A...), A)
Hcat(args...) = _Hcat(args)

copy(x::Hcat) = Hcat(copy.(x.arrays)...)

size(f::Hcat) = (size(f.arrays[1],1), +(map(a -> size(a,2), f.arrays)...))
Base.IndexStyle(::Type{<:Hcat}) where T = Base.IndexCartesian()

function getindex(f::Hcat{T}, k::Integer, j::Integer) where T
    ξ = j
    for A in f.arrays
        n = size(A,2)
        ξ ≤ n && return T(A[k,ξ])::T
        ξ -= n
    end
    throw(BoundsError(f, (k,j)))
end

function setindex!(f::Hcat{T}, v, k::Integer, j::Integer) where T
    ξ = j
    for A in f.arrays
        n = size(A,2)
        ξ ≤ n && return setindex!(A, v, k, ξ)
        ξ -= n
    end
    throw(BoundsError(f, (k,j)))
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
            dest[:, pos:p1] .= a
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


#####
# adjoint/transpose
#####

for adj in (:adjoint, :transpose)
    @eval begin
        $adj(A::Hcat{T}) where T = _Vcat(T, $adj.(A.arrays))
        $adj(A::Vcat{T}) where T = _Hcat(T, $adj.(A.arrays))
    end
end

_vec(a) = a
_vec(a::AbstractArray) = vec(a)

vec(A::Hcat) = Vcat(_vec.(A.arrays)...)


#####
# broadcasting
#
# We want broadcasting for numbers with concaenations to pass through
# to take advantage of special implementations of the sub-components
######

BroadcastStyle(::Type{<:AbstractConcatArray{<:Any,N}}) where N = LazyArrayStyle{N}()

broadcasted(::LazyArrayStyle, op, A::Vcat) =
    _Vcat(broadcast(x -> broadcast(op, x), A.arrays))

broadcasted(::LazyArrayStyle, op, A::Vcat, c::Number) =
    _Vcat(broadcast((x,y) -> broadcast(op, x, y), A.arrays, c))
broadcasted(::LazyArrayStyle, op, c::Number, A::Vcat) =
    _Vcat(broadcast((x,y) -> broadcast(op, x, y), c, A.arrays))
broadcasted(::LazyArrayStyle, op, A::Vcat, c::Ref) =
    _Vcat(broadcast((x,y) -> broadcast(op, x, Ref(y)), A.arrays, c))
broadcasted(::LazyArrayStyle, op, c::Ref, A::Vcat) =
    _Vcat(broadcast((x,y) -> broadcast(op, Ref(x), y), c, A.arrays))    


# determine indices of components of a vcat
_vcat_axes(::Tuple{}) = (1,)
_vcat_axes(a::Tuple{<:AbstractUnitRange}) = (first(a),)
_vcat_axes(::Tuple{}, b, c...) = tuple(1, broadcast(x -> broadcast(+, 1, x), _vcat_axes(b, c...))...)
_vcat_axes(a::Tuple{<:AbstractUnitRange}, b, c...) = tuple(first(a), broadcast((α,x) -> broadcast(+, α, x), last(first(a)),
                                                            _vcat_axes(b, c...))...)

_vcat_getindex_eval(y) = ()
_vcat_getindex_eval(y, a, b...) = tuple(y[a], _vcat_getindex_eval(y, b...)...)

function broadcasted(::LazyArrayStyle, op, A::Vcat{<:Any,1}, B::AbstractVector)
    kr = _vcat_axes(axes.(A.arrays)...)  # determine how to break up B
    B_arrays = _vcat_getindex_eval(B,kr...)    # evaluate B at same chunks as A
    _Vcat(broadcast((a,b) -> broadcast(op,a,b), A.arrays, B_arrays))
end

function broadcasted(::LazyArrayStyle, op, A::AbstractVector, B::Vcat{<:Any,1})
    kr = _vcat_axes(axes.(B.arrays)...)
    A_arrays = _vcat_getindex_eval(A,kr...)
    _Vcat(broadcast((a,b) -> broadcast(op,a,b), A_arrays, B.arrays))
end

# Cannot broadcast Vcat's in a lazy way so stick to BroadcastArray
broadcasted(::LazyArrayStyle, op, A::Vcat{<:Any,1}, B::Vcat{<:Any,1}) =
    Broadcasted{LazyArrayStyle}(op, (A, B))


function +(A::Vcat, B::Vcat)
    size(A) == size(B) || throw(DimensionMismatch("dimensions must match."))
    A .+ B
end
function +(A::Vcat, B::AbstractArray)
    size(A) == size(B) || throw(DimensionMismatch("dimensions must match."))
    A .+ B
end
function +(A::AbstractArray, B::Vcat)
    size(A) == size(B) || throw(DimensionMismatch("dimensions must match."))
    A .+ B
end

######
# Special Vcat broadcasts
#
# We use Vcat for infinite padded vectors, so we need to special case
# two arrays. This may be generalisable in the future
######

function _vcat_broadcasted(::Type{T}, op, (Ahead, Atail)::Tuple{<:AbstractVector,<:AbstractFill},
                               (Bhead, Btail)::Tuple{<:AbstractVector,<:AbstractFill}) where T
    if length(Ahead) ≥ length(Bhead)
        M,m = length(Ahead), length(Bhead)
        Chead = Vector{T}(undef,M)
        view(Chead,1:m) .= op.(view(Ahead,1:m), Bhead)
        view(Chead,m+1:M) .= op.(view(Ahead,m+1:M),Btail[1:M-m])

        Ctail = op.(Atail, Btail[M-m+1:end])
    else
        m,M = length(Ahead), length(Bhead)
        Chead = Vector{T}(undef,M)
        view(Chead,1:m) .= op.(Ahead, view(Bhead,1:m))
        view(Chead,m+1:M) .= op.(Atail[1:M-m],view(Bhead,m+1:M))

        Ctail = op.(Atail[M-m+1:end], Btail)
    end

    _Vcat((Chead, Ctail))
end

_vcat_broadcasted(::Type{T}, op, (Ahead, Atail)::Tuple{<:Number,<:AbstractFill},
                           (Bhead, Btail)::Tuple{<:Number,<:AbstractFill}) where {M,T} =
   _Vcat((op.(Ahead,Bhead), op.(Atail,Btail)))

_vcat_broadcasted(::Type{T}, op, (Ahead, Atail)::Tuple{<:SVector{M},<:AbstractFill},
                           (Bhead, Btail)::Tuple{<:SVector{M},<:AbstractFill}) where {M,T} =
   _Vcat((op.(Ahead,Bhead), op.(Atail,Btail)))

# default is BroadcastArray
_vcat_broadcasted(::Type{T}, op, A, B) where T =
    Broadcasted{LazyArrayStyle}(op, (_Vcat(A), _Vcat(B)))


broadcasted(::LazyArrayStyle{1}, op, A::Vcat{T, 1, <:Tuple{<:Any,<:Any}},
                                     B::Vcat{V, 1, <:Tuple{<:Any,<:Any}}) where {T,V} =
  _vcat_broadcasted(promote_type(T,V), op, A.arrays, B.arrays)



####
# Cumsum
####

sum(V::Vcat) = mapreduce(sum, +, V.arrays)
all(V::Vcat) = all(all.(V.arrays))
any(V::Vcat) = any(any.(V.arrays))
all(f::Function, V::Vcat) = all(all.(f, V.arrays))
any(f::Function, V::Vcat) = any(any.(f, V.arrays))

_dotplus(a,b) = broadcast(+, a, b)

@inline _cumsum(x::Number) = x
@inline _cumsum(x) = cumsum(x)
@generated function _vcat_cumsum(x...)
    N = length(x)
    ret = quote
        @nexprs $N d->(c_d = _cumsum(x[d]))
        d_1 = c_1
        @nexprs $(N-1) k->(d_{k+1} = broadcast(+, last(d_k), c_{k+1}))
        @ntuple $N d
    end
end

@inline cumsum(V::Vcat{<:Any,1}) = _Vcat(_vcat_cumsum(V.arrays...))


_vcat_diff(x::Number) = ()
_vcat_diff(x) = (diff(x),)

_vcat_diff(a::Number, b, c...) = (first(b)-a, _vcat_diff(b,c...)...)
_vcat_diff(a, b, c...) = (diff(a), first(b)-last(a), _vcat_diff(b,c...)...)
@inline diff(V::Vcat{T,1}) where T = _Vcat(T,_vcat_diff(V.arrays...))

####
# maximum/minimum
####

for op in (:maximum, :minimum)
    @eval $op(V::Vcat) = $op($op.(V.arrays))
end

function in(x, V::Vcat) 
    for a in V.arrays
        in(x, a) && return true
    end
    false
end

_fill!(a, x) = fill!(a,x)
function _fill!(a::Number, x) 
    a == x || throw(ArgumentError("Cannot set $a to $x"))
    a
end

function fill!(V::AbstractConcatArray, x) 
    for a in V.arrays
        _fill!(a, x)
    end
    V
end

###
# *
###

struct VcatLayout{Lays} <: MemoryLayout end
struct HcatLayout{Lays} <: MemoryLayout end

tuple_type_memorylayouts(::Type{I}) where I<:Tuple = Tuple{typeof.(MemoryLayout.(I.parameters))...}
tuple_type_memorylayouts(::Type{Tuple{A}}) where {A} = Tuple{typeof(MemoryLayout(A))}
tuple_type_memorylayouts(::Type{Tuple{A,B}}) where {A,B} = Tuple{typeof(MemoryLayout(A)),typeof(MemoryLayout(B))}
tuple_type_memorylayouts(::Type{Tuple{A,B,C}}) where {A,B,C} = Tuple{typeof(MemoryLayout(A)),typeof(MemoryLayout(B)),typeof(MemoryLayout(C))}


MemoryLayout(::Type{Vcat{T,N,I}}) where {T,N,I} = VcatLayout{tuple_type_memorylayouts(I)}()
MemoryLayout(::Type{Hcat{T,I}}) where {T,I} = HcatLayout{tuple_type_memorylayouts(I)}()

@lazymul Hcat
@lazymul Vcat{<:Any,2}

*(A::Hcat, B::Vcat{<:Any,2}) = materialize(Mul(A,B))
*(A::Vcat{<:Any,2}, B::Hcat) = materialize(Mul(A,B))

function materialize!(M::MatMulVecAdd{<:HcatLayout,<:VcatLayout})
    α,A,B,β,C =  M.α,M.A,M.B,M.β,M.C
    T = eltype(C)
    _fill_lmul!(β,C) # this is temporary until strong β = false is supported
    for (a,b) in zip(A.arrays,B.arrays)
        materialize!(MulAdd(α,a,b,one(T),C))
    end
    C
 end

 function materialize!(M::MatMulMatAdd{<:HcatLayout,<:VcatLayout})
    α,A,B,β,C =  M.α,M.A,M.B,M.β,M.C
    T = eltype(C)
    _fill_lmul!(β,C) # this is temporary until strong β = false is supported
    for (a,b) in zip(A.arrays,B.arrays)
        materialize!(MulAdd(α,a,b,one(T),C))
    end
    C
 end