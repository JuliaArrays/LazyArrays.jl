# Lazy concatenation of AbstractVector's.
# Similar to Iterators.Flatten and some code has been reused from julia/base/iterators.jl


# This fixes type inference issues in v1.0
for op in  (:hcat, :vcat)
    @eval call(::ApplyLayout{typeof($op)}, _) = $op
end


const Vcat{T,N,I<:Tuple} = ApplyArray{T,N,typeof(vcat),I}

Vcat(A...) = ApplyArray(vcat, A...)
Vcat{T}(A...) where T = ApplyArray{T}(vcat, A...)
Vcat() = Vcat{Any}()

Vcat(A::AbstractVector...) = ApplyVector(vcat, A...)
Vcat{T}(A::AbstractVector...) where T = ApplyVector{T}(vcat, A...)

function instantiate(A::Applied{DefaultApplyStyle,typeof(vcat)})
    isempty(A.args) && return A
    m = size(A.args[1],2)
    for k=2:length(A.args)
        size(A.args[k],2) == m || throw(ArgumentError("number of columns of each array must match (got $(map(x->size(x,2), A)))"))
    end
    Applied{DefaultApplyStyle}(A.f,map(instantiate,A.args))
end

@inline eltype(A::Applied{<:Any,typeof(vcat)}) = promote_type(map(eltype,A.args)...)
@inline eltype(A::Applied{<:Any,typeof(vcat),Tuple{}}) = Any
@inline ndims(A::Applied{<:Any,typeof(vcat),I}) where I = max(1,maximum(map(ndims,A.args)))
@inline ndims(A::Applied{<:Any,typeof(vcat),Tuple{}}) = 1
@inline axes(f::Vcat{<:Any,1,Tuple{}}) = (OneTo(0),)
@inline axes(f::Vcat{<:Any,1}) = tuple(OneTo(+(map(length,f.args)...)))
@inline axes(f::Vcat{<:Any,2}) = (OneTo(+(map(a -> size(a,1), f.args)...)), axes(f.args[1],2))
Base.IndexStyle(::Type{<:Vcat{T,1}}) where T = Base.IndexLinear()
Base.IndexStyle(::Type{<:Vcat{T,2}}) where T = Base.IndexCartesian()

function ==(a::Vcat{T,1,II}, b::Vcat{T,1,II}) where {T,II}
    if !all(map(length,arguments(a)) .== map(length,arguments(b)))
        return Base.invoke(==, NTuple{2,AbstractArray}, a, b)
    end
    all(arguments(a) .== arguments(b))
end

@propagate_inbounds @inline vcat_getindex(f, idx::Vararg{Integer}) =
    vcat_getindex_recursive(f, idx, f.args...)

@propagate_inbounds @inline function vcat_getindex_recursive(
        f, idx::NTuple{1}, A, args...)
    k, = idx
    T = eltype(f)
    n = length(A)
    k ≤ n && return convert(T, A[k])::T
    vcat_getindex_recursive(f, (k - n, ), args...)
end

@propagate_inbounds @inline function vcat_getindex_recursive(
        f, idx::NTuple{2}, A, args...)
    k, j = idx
    T = eltype(f)
    n = size(A, 1)
    k ≤ n && return convert(T, A[k, j])::T
    vcat_getindex_recursive(f, (k - n, j), args...)
end

@inline vcat_getindex_recursive(f, idx) = throw(BoundsError(f, idx))

@propagate_inbounds @inline getindex(f::Vcat{<:Any,1}, k::Integer) = vcat_getindex(f, k)
@propagate_inbounds @inline getindex(f::Vcat{<:Any,2}, k::Integer, j::Integer) = vcat_getindex(f, k, j)
getindex(f::Applied{DefaultArrayApplyStyle,typeof(vcat)}, k::Integer)= vcat_getindex(f, k)
getindex(f::Applied{DefaultArrayApplyStyle,typeof(vcat)}, k::Integer, j::Integer)= vcat_getindex(f, k, j)
getindex(f::Applied{<:Any,typeof(vcat)}, k::Integer)= vcat_getindex(f, k)
getindex(f::Applied{<:Any,typeof(vcat)}, k::Integer, j::Integer)= vcat_getindex(f, k, j)


# since its mutable we need to make a copy
copy(f::Vcat) = Vcat(map(copy, f.args)...)
map(::typeof(copy), f::Vcat) = Vcat(map.(copy, f.args)...)

@propagate_inbounds @inline vcat_setindex!(f, v, idx::Vararg{Integer}) =
    vcat_setindex_recursive!(f, v, idx, f.args...)

@propagate_inbounds @inline function vcat_setindex_recursive!(
        f, v, idx::NTuple{1}, A, args...)
    k, = idx
    n = length(A)
    k ≤ n && return setindex!(A, v, idx...)
    vcat_setindex_recursive!(f, v, (k - n, ), args...)
end

@propagate_inbounds @inline function vcat_setindex_recursive!(
        f, v, idx::NTuple{2}, A, args...)
    k, j = idx
    n = size(A, 1)
    k ≤ n && return setindex!(A, v, idx...)
    vcat_setindex_recursive!(f, v, (k - n, j), args...)
end

@inline vcat_setindex_recursive!(f, v, idx) = throw(BoundsError(f, idx))

@propagate_inbounds @inline function setindex!(
        f::Vcat{T,N}, v, idx::Vararg{Integer,N}) where {T,N}
    vcat_setindex_recursive!(f, v, idx, f.args...)
end

reverse(f::Vcat{<:Any,1}) = Vcat((reverse(itr) for itr in reverse(f.args))...)


const Hcat{T,I<:Tuple} = ApplyArray{T,2,typeof(hcat),I}

Hcat(A...) = ApplyArray(hcat, A...)
Hcat{T}(A...) where T = ApplyArray{T}(hcat, A...)

function instantiate(A::Applied{DefaultApplyStyle,typeof(hcat)})
    isempty(A.args) && return A
    m = size(A.args[1],1)
    for k=2:length(A.args)
        size(A.args[k],1) == m || throw(ArgumentError("number of rows of each array must match (got $(map(x->size(x,1), A)))"))
    end
    Applied{DefaultApplyStyle}(A.f,map(instantiate,A.args))
end

@inline eltype(A::Applied{<:Any,typeof(hcat)}) = promote_type(map(eltype,A.args)...)
ndims(::Applied{<:Any,typeof(hcat)}) = 2
size(f::Applied{<:Any,typeof(hcat)}) = (size(f.args[1],1), +(map(a -> size(a,2), f.args)...))
Base.IndexStyle(::Type{<:Hcat}) where T = Base.IndexCartesian()

@inline hcat_getindex(f, k::Integer, j::Integer) =
    hcat_getindex_recursive(f, (k, j), f.args...)

@inline function hcat_getindex_recursive(
        f, idx::NTuple{2}, A, args...)
    k, j = idx
    T = eltype(f)
    n = size(A, 2)
    j ≤ n && return convert(T, A[k, j])::T
    hcat_getindex_recursive(f, (k, j - n), args...)
end

@inline hcat_getindex_recursive(f, idx) = throw(BoundsError(f, idx))

getindex(f::Hcat, k::Integer, j::Integer) = hcat_getindex(f, k, j)
getindex(f::Applied{DefaultArrayApplyStyle,typeof(hcat)}, k::Integer, j::Integer)= hcat_getindex(f, k, j)
getindex(f::Applied{<:Any,typeof(hcat)}, k::Integer, j::Integer)= hcat_getindex(f, k, j)

# since its mutable we need to make a copy
copy(f::Hcat) = Hcat(map(copy, f.args)...)

@inline function hcat_setindex_recursive!(
        f, v, idx::NTuple{2}, A, args...)
    k, j = idx
    T = eltype(f)
    n = size(A, 2)
    j ≤ n && return setindex!(A, v, k, j)
    hcat_setindex_recursive!(f, v, (k, j - n), args...)
end

@inline hcat_setindex_recursive!(f, v, idx) = throw(BoundsError(f, idx))

function setindex!(f::Hcat{T}, v, k::Integer, j::Integer) where T
    hcat_setindex_recursive!(f, v, (k, j), f.args...)
end


## copyto!
# based on Base/array.jl, Base/abstractarray.jl
_copyto!(_, LAY::ApplyLayout{typeof(vcat)}, dest::AbstractArray{<:Any,N}, V::AbstractArray{<:Any,N}) where N =
    vcat_copyto!(dest, arguments(LAY, V)...)
function vcat_copyto!(dest::AbstractMatrix, arrays...)
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
        copyto!(view(dest,pos:p1, :), a)
        pos = p1+1
    end
    return dest
end

function vcat_copyto!(arr::AbstractVector, arrays...)
    n = 0
    for a in arrays
        n += length(a)
    end
    n == length(arr) || throw(DimensionMismatch("destination must have length equal to sums of concatenated vectors"))

    i = 0
    @inbounds for a in arrays
        m = length(a)
        copyto!(view(arr,i+1:i+m), a)
        i += m
    end
    arr
end

function vcat_copyto!(arr::Vector{T}, arrays::Vector{T}...) where T
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

_copyto!(_, LAY::ApplyLayout{typeof(hcat)}, dest::AbstractMatrix, H::AbstractMatrix) =
    hcat_copyto!(dest, arguments(LAY,H)...)
function hcat_copyto!(dest::AbstractMatrix, arrays...)
    nargs = length(arrays)
    nrows = size(dest, 1)
    ncols = 0
    dense = true
    for a in arrays
        dense &= isa(a,Array)
        nd = ndims(a)
        ncols += (nd==2 ? size(a,2) : 1)
    end

    nrows == size(first(arrays),1) || throw(DimensionMismatch("Destination rows must match"))
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
            copyto!(view(dest,:, pos:p1), a)
            pos = p1+1
        end
    end
    return dest
end

function hcat_copyto!(dest::AbstractMatrix, arrays::AbstractVector...)
    height = size(dest, 1)
    for j = 1:length(arrays)
        if length(arrays[j]) != height
            throw(DimensionMismatch("vectors must have same lengths"))
        end
    end
    for j=1:length(arrays)
        copyto!(view(dest,:,j), arrays[j])
    end

    dest
end


#####
# adjoint/transpose
#####

for adj in (:adjoint, :transpose)
    @eval begin
        $adj(A::Hcat{T}) where T = Vcat{T}(map($adj,A.args)...)
        $adj(A::Vcat{T}) where T = Hcat{T}(map($adj,A.args)...)
    end
end

_vec(a) = a
_vec(a::AbstractArray) = vec(a)
_vec(a::Adjoint{<:Number,<:AbstractVector}) = _vec(parent(a))
vec(A::Hcat) = Vcat(map(_vec,A.args)...)

_permutedims(a) = a
_permutedims(a::AbstractArray) = permutedims(a)

permutedims(A::Hcat{T}) where T = Vcat{T}(map(_permutedims,A.args)...)
permutedims(A::Vcat{T}) where T = Hcat{T}(map(_permutedims,A.args)...)


#####
# broadcasting
#
# We want broadcasting for numbers with concaenations to pass through
# to take advantage of special implementations of the sub-components
######

BroadcastStyle(::Type{<:Vcat{<:Any,N}}) where N = LazyArrayStyle{N}()
BroadcastStyle(::Type{<:Hcat{<:Any}}) where N = LazyArrayStyle{2}()

broadcasted(::LazyArrayStyle, op, A::Vcat) =
    Vcat(broadcast(x -> broadcast(op, x), A.args)...)

for Cat in (:Vcat, :Hcat)
    @eval begin
        broadcasted(::LazyArrayStyle, op, A::$Cat, c::Number) =
            $Cat(broadcast((x,y) -> broadcast(op, x, y), A.args, c)...)
        broadcasted(::LazyArrayStyle, op, c::Number, A::$Cat) =
            $Cat(broadcast((x,y) -> broadcast(op, x, y), c, A.args)...)
        broadcasted(::LazyArrayStyle, op, A::$Cat, c::Ref) =
            $Cat(broadcast((x,y) -> broadcast(op, x, Ref(y)), A.args, c)...)
        broadcasted(::LazyArrayStyle, op, c::Ref, A::$Cat) =
            $Cat(broadcast((x,y) -> broadcast(op, Ref(x), y), c, A.args)...)
    end
end


# determine indices of components of a vcat
_vcat_axes(::Tuple{}) = (1,)
_vcat_axes(a::Tuple{<:AbstractUnitRange}) = (first(a),)
_vcat_axes(::Tuple{}, b, c...) = tuple(1, broadcast(x -> broadcast(+, 1, x), _vcat_axes(b, c...))...)
_vcat_axes(a::Tuple{<:AbstractUnitRange}, b, c...) = tuple(first(a), broadcast((α,x) -> broadcast(+, α, x), last(first(a)),
                                                            _vcat_axes(b, c...))...)

_vcat_getindex_eval(y) = ()
_vcat_getindex_eval(y, a, b...) = tuple(y[a], _vcat_getindex_eval(y, b...)...)

function broadcasted(::LazyArrayStyle, op, A::Vcat{<:Any,1}, B::AbstractVector)
    kr = _vcat_axes(map(axes,A.args)...)  # determine how to break up B
    B_arrays = _vcat_getindex_eval(B,kr...)    # evaluate B at same chunks as A
    ApplyVector(vcat, broadcast((a,b) -> broadcast(op,a,b), A.args, B_arrays)...)
end

function broadcasted(::LazyArrayStyle, op, A::AbstractVector, B::Vcat{<:Any,1})
    kr = _vcat_axes(axes.(B.args)...)
    A_arrays = _vcat_getindex_eval(A,kr...)
    Vcat(broadcast((a,b) -> broadcast(op,a,b), A_arrays, B.args)...)
end

# Cannot broadcast Vcat's in a lazy way so stick to BroadcastArray
broadcasted(::LazyArrayStyle, op, A::Vcat{<:Any,1}, B::Vcat{<:Any,1}) =
    Broadcasted{LazyArrayStyle}(op, (A, B))

# ambiguities
broadcasted(::LazyArrayStyle, op, A::Vcat{<:Any,1}, B::CachedVector) = cache_broadcast(op, A, B)
broadcasted(::LazyArrayStyle, op, A::CachedVector, B::Vcat{<:Any,1}) = cache_broadcast(op, A, B)

broadcasted(::LazyArrayStyle{1}, ::typeof(*), a::Vcat{<:Any,1}, b::Zeros{<:Any,1})=
    broadcast(DefaultArrayStyle{1}(), *, a, b)

broadcasted(::LazyArrayStyle{1}, ::typeof(*), a::Zeros{<:Any,1}, b::Vcat{<:Any,1})=
    broadcast(DefaultArrayStyle{1}(), *, a, b)



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

    Vcat(Chead, Ctail)
end

_vcat_broadcasted(::Type{T}, op, (Ahead, Atail)::Tuple{<:Number,<:AbstractFill},
                           (Bhead, Btail)::Tuple{<:Number,<:AbstractFill}) where {M,T} =
   Vcat(op.(Ahead,Bhead), op.(Atail,Btail))

_vcat_broadcasted(::Type{T}, op, (Ahead, Atail)::Tuple{<:SVector{M},<:AbstractFill},
                           (Bhead, Btail)::Tuple{<:SVector{M},<:AbstractFill}) where {M,T} =
   Vcat(op.(Ahead,Bhead), op.(Atail,Btail))

# default is BroadcastArray
_vcat_broadcasted(::Type{T}, op, A, B) where T =
    Broadcasted{LazyArrayStyle}(op, (Vcat(A...), Vcat(B...)))


broadcasted(::LazyArrayStyle{1}, op, A::Vcat{T, 1, <:Tuple{<:Any,<:Any}},
                                     B::Vcat{V, 1, <:Tuple{<:Any,<:Any}}) where {T,V} =
  _vcat_broadcasted(promote_type(T,V), op, A.args, B.args)



####
# Cumsum
####

sum(V::Vcat) = mapreduce(sum, +, V.args)
all(V::Vcat) = all(all.(V.args))
any(V::Vcat) = any(any.(V.args))
all(f::Function, V::Vcat) = all(all.(f, V.args))
any(f::Function, V::Vcat) = any(any.(f, V.args))

_dotplus(a,b) = broadcast(+, a, b)

@inline _cumsum(x::Number) = x
@inline _cumsum(x) = cumsum(x)
_cumsum_last(x::AbstractVector{T}) where T = isempty(x) ? zero(T) : last(x)
_cumsum_last(x) = last(x)

_tuple_cumsum() = ()
_tuple_cumsum(a) = (a,)
_tuple_cumsum(a, b...) = (a, broadcast(+,a,_tuple_cumsum(b...))...)
function _vcat_cumsum(x...)
    cs = map(_cumsum,x)
    cslasts = tuple(0,_tuple_cumsum(map(_cumsum_last,most(cs))...)...)
    map((a,b) -> broadcast(+,a,b), cslasts, cs)
end

@inline cumsum(V::Vcat{<:Any,1}) = ApplyVector(vcat,_vcat_cumsum(V.args...)...)


_vcat_diff(x::Number) = ()
_vcat_diff(x) = (diff(x),)

_vcat_diff(a::Number, b, c...) = (first(b)-a, _vcat_diff(b,c...)...)
_vcat_diff(a, b, c...) = (diff(a), first(b)-last(a), _vcat_diff(b,c...)...)
@inline diff(V::Vcat{T,1}) where T = ApplyVector{T}(vcat,_vcat_diff(V.args...)...)

####
# maximum/minimum
####

for op in (:maximum, :minimum)
    @eval $op(V::Vcat) = $op($op.(V.args))
end

function in(x, V::Vcat)
    for a in V.args
        in(x, a) && return true
    end
    false
end

_fill!(a, x) = fill!(a,x)
function _fill!(a::Number, x)
    a == x || throw(ArgumentError("Cannot set $a to $x"))
    a
end

function fill!(V::Union{Vcat,Hcat}, x)
    for a in V.args
        _fill!(a, x)
    end
    V
end

###
# *
###

function materialize!(M::MatMulVecAdd{ApplyLayout{typeof(hcat)},ApplyLayout{typeof(vcat)}})
    α,A,B,β,C =  M.α,M.A,M.B,M.β,M.C
    T = eltype(C)
    _fill_lmul!(β,C) # this is temporary until strong β = false is supported
    for (a,b) in zip(A.args,B.args)
        materialize!(MulAdd(α,a,b,one(T),C))
    end
    C
 end

 function materialize!(M::MatMulMatAdd{ApplyLayout{typeof(hcat)},ApplyLayout{typeof(vcat)}})
    α,A,B,β,C =  M.α,M.A,M.B,M.β,M.C
    T = eltype(C)
    _fill_lmul!(β,C) # this is temporary until strong β = false is supported
    for (a,b) in zip(A.args,B.args)
        materialize!(MulAdd(α,a,b,one(T),C))
    end
    C
 end

 ####
 # col/rowsupport
 ####


most(a) = reverse(tail(reverse(a)))
colsupport(M::Vcat, j) = first(colsupport(first(M.args),j)):(size(Vcat(most(M.args)...),1)+last(colsupport(last(M.args),j)))

function rowsupport(V::Vcat, k::Integer)
    ξ = k
    for A in arguments(V)
        n = size(A,1)
        ξ ≤ n && return rowsupport(A, ξ)
        ξ -= n
    end
    return 1:0
end

function colsupport(H::Hcat, j::Integer)
    ξ = j
    for A in arguments(H)
        n = size(A,2)
        ξ ≤ n && return colsupport(A, ξ)
        ξ -= n
    end
    return 1:0
end

rowsupport(M::Hcat, k) = first(rowsupport(first(M.args),k)):(size(Hcat(most(M.args)...),2)+last(rowsupport(last(M.args),k)))


###
# padded
####

struct PaddedLayout{L} <: MemoryLayout end
applylayout(::Type{typeof(vcat)}, ::A, ::ZerosLayout) where A = PaddedLayout{A}()
applylayout(::Type{typeof(vcat)}, ::ScalarLayout, ::ScalarLayout, ::ZerosLayout) = PaddedLayout{ApplyLayout{typeof(vcat)}}()
applylayout(::Type{typeof(vcat)}, ::A, ::PaddedLayout) where A = PaddedLayout{ApplyLayout{typeof(vcat)}}()
applylayout(::Type{typeof(vcat)}, ::ScalarLayout, ::ScalarLayout, ::PaddedLayout) = PaddedLayout{ApplyLayout{typeof(vcat)}}()
cachedlayout(::A, ::ZerosLayout) where A = PaddedLayout{A}()
sublayout(::PaddedLayout{Lay}, sl::Type{<:Tuple{Slice,Integer}}) where Lay =
    PaddedLayout{typeof(sublayout(Lay(), sl))}()

paddeddata(A::CachedArray) = view(A.data,OneTo.(A.datasize)...)
_vcat_paddeddata(A, B::Zeros) = A
_vcat_paddeddata(A, B) = Vcat(A, paddeddata(B))
_vcat_paddeddata(A, B, C...) = Vcat(A, _vcat_paddeddata(B, C...))
paddeddata(A::Vcat) = _vcat_paddeddata(A.args...)

colsupport(::PaddedLayout, A, j) = colsupport(paddeddata(A),j)

function _vcat_resizedata!(::PaddedLayout, B, m)
    m ≤ length(paddeddata(B))  || throw(ArgumentError("Cannot resize"))
    B
end
resizedata!(B::Vcat, m) = _vcat_resizedata!(MemoryLayout(B), B, m)

function ==(A::CachedVector{<:Any,<:Any,<:Zeros}, B::CachedVector{<:Any,<:Any,<:Zeros})
    length(A) == length(B) || return false
    n = max(A.datasize[1], B.datasize[1])
    resizedata!(A,n); resizedata!(B,n)
    view(A.data,OneTo(n)) == view(B.data,OneTo(n))
end

# special copyto! since `similar` of a padded returns a cached
for Typ in (:Number, :AbstractVector)
    @eval begin
        function _copyto!(::PaddedLayout, ::PaddedLayout, dest::CachedVector{<:$Typ}, src::AbstractVector)
            length(src) ≤ length(dest)  || throw(BoundsError())
            a = paddeddata(src)
            n = length(a)
            resizedata!(dest, n) # make sure we are padded enough
            copyto!(view(dest.data,OneTo(n)), a)
            dest
        end
        function _copyto!(::PaddedLayout, ::PaddedLayout, dest::SubArray{<:Any,1,<:CachedVector{<:$Typ}}, src::AbstractVector)
            length(src) ≤ length(dest)  || throw(BoundsError())
            a = paddeddata(src)
            n = length(a)
            k = first(parentindices(dest)[1])
            resizedata!(parent(dest), k+n-1) # make sure we are padded enough
            copyto!(view(parent(dest).data,k:k+n-1), a)
            dest
        end

        _copyto!(::PaddedLayout, ::PaddedLayout, dest::CachedVector{<:$Typ}, src::CachedVector) =
            _padded_copyto!(dest, src)
    end
end

function _padded_copyto!(dest::CachedVector, src::CachedVector)
    length(src) ≤ length(dest)  || throw(BoundsError())
    n = src.datasize[1]
    resizedata!(dest, n)
    copyto!(view(dest.data,OneTo(n)), view(src.data,OneTo(n)))
    dest
end

_copyto!(::PaddedLayout, ::PaddedLayout, dest::CachedVector, src::CachedVector) =
    _padded_copyto!(dest, src)

function _copyto!(::PaddedLayout, ::ZerosLayout, dest::AbstractVector, src::AbstractVector)
    zero!(paddeddata(dest))
    dest
end

# special case handle broadcasting with padded and cached arrays
function _cache_broadcast(::PaddedLayout, ::PaddedLayout, op, A, B)
    a,b = paddeddata(A),paddeddata(B)
    n,m = length(a),length(b)
    dat = if n ≤ m
        [broadcast(op, a, view(b,1:n)); broadcast(op, zero(eltype(A)), @view(b[n+1:end]))]
    else
        [broadcast(op, view(a,1:m), b); broadcast(op, @view(a[m+1:end]), zero(eltype(B)))]
    end
    CachedArray(dat, broadcast(op, Zeros{eltype(A)}(length(A)), Zeros{eltype(B)}(length(B))))
end

function _cache_broadcast(_, ::PaddedLayout, op, A, B)
    b = paddeddata(B)
    m = length(b)
    zB = Zeros{eltype(B)}(size(B)...)
    CachedArray(broadcast(op, view(A,1:m), b), broadcast(op, A, zB))
end

function _cache_broadcast(::PaddedLayout, _, op, A, B)
    a = paddeddata(A)
    n = length(a)
    zA = Zeros{eltype(A)}(size(A)...)
    CachedArray(broadcast(op, a, view(B,1:n)), broadcast(op, zA, B))
end

function _cache_broadcast(::PaddedLayout, ::CachedLayout, op, A, B)
    a,b = paddeddata(A),paddeddata(B)
    n = length(a)
    resizedata!(B,n)
    Bdata = paddeddata(B)
    b = view(Bdata,1:n)
    zA1 = Zeros{eltype(A)}(size(Bdata,1)-n)
    zA = Zeros{eltype(A)}(size(A)...)
    CachedArray([broadcast(op, a, b); broadcast(op, zA1, @view(Bdata[n+1:end]))], broadcast(op, zA, B.array))
end

function _cache_broadcast(::CachedLayout, ::PaddedLayout, op, A, B)
    b = paddeddata(B)
    n = length(b)
    resizedata!(A,n)
    Adata = paddeddata(A)
    a = view(Adata,1:n)
    zB1 = Zeros{eltype(B)}(size(Adata,1)-n)
    zB = Zeros{eltype(B)}(size(B)...)
    CachedArray([broadcast(op, a, b); broadcast(op, @view(Adata[n+1:end]), zB1)], broadcast(op, A.array, zB))
end


###
# Dot/Axpy
###


struct Axpy{StyleX,StyleY,T,XTyp,YTyp}
    α::T
    X::XTyp
    Y::YTyp
end

Axpy(α::T, X::XTyp, Y::YTyp) where {T,XTyp,YTyp} = Axpy{typeof(MemoryLayout(XTyp)), typeof(MemoryLayout(YTyp)), T, XTyp, YTyp}(α, X, Y)
materialize!(d::Axpy{<:Any,<:Any,<:Number,<:AbstractArray,<:AbstractArray}) = Base.invoke(BLAS.axpy!, Tuple{Number,AbstractArray,AbstractArray}, d.α, d.X, d.Y)
function materialize!(d::Axpy{<:PaddedLayout,<:PaddedLayout,U,<:AbstractVector{T},<:AbstractVector{V}}) where {U,T,V}
    x = paddeddata(d.X)
    resizedata!(d.Y, length(x))
    y = paddeddata(d.Y)
    BLAS.axpy!(d.α, x, view(y,1:length(x)))
    y
end
axpy!(α, X, Y) = materialize!(Axpy(α,X,Y))
BLAS.axpy!(α, X::LazyArray, Y::AbstractArray) = axpy!(α, X, Y)
BLAS.axpy!(α, X::SubArray{<:Any,N,<:LazyArray}, Y::AbstractArray) where N = axpy!(α, X, Y)


###
# l/rmul!
###

function materialize!(M::Lmul{ScalarLayout,<:PaddedLayout})
    lmul!(M.A, paddeddata(M.B))
    M.B
end

function materialize!(M::Rmul{<:PaddedLayout,ScalarLayout})
    rmul!(paddeddata(M.A), M.B)
    M.A
end

###
# norm
###

for Cat in (:Vcat, :Hcat)
    for (op,p) in ((:norm1,1), (:norm2,2), (:normInf,Inf))
        @eval $op(a::$Cat) = $op(norm.(a.args,$p))
    end
    @eval normp(a::$Cat, p) = norm(norm.(a.args, p), p)
end

_norm2(::PaddedLayout, a) = norm(paddeddata(a),2)
_norm1(::PaddedLayout, a) = norm(paddeddata(a),1)
_normInf(::PaddedLayout, a) = norm(paddeddata(a),Inf)
_normp(::PaddedLayout, a, p) = norm(paddeddata(a),p)


function copy(D::Dot{<:PaddedLayout, <:PaddedLayout})
    a = paddeddata(D.A)
    b = paddeddata(D.B)
    m = min(length(a), length(b))
    convert(eltype(D), dot(view(a, 1:m), view(b, 1:m)))
end

function copy(D::Dot{<:PaddedLayout})
    a = paddeddata(D.A)
    m = length(a)
    convert(eltype(D), dot(a, view(D.B, 1:m)))
end

function copy(D::Dot{<:Any, <:PaddedLayout})
    b = paddeddata(D.B)
    m = length(b)
    convert(eltype(D), dot(view(D.A, 1:m), b))
end



###
# subarrays
###

sublayout(::ApplyLayout{typeof(vcat)}, _) = ApplyLayout{typeof(vcat)}()
sublayout(::ApplyLayout{typeof(hcat)}, _) = ApplyLayout{typeof(hcat)}()
# a row-slice of an Hcat is equivalent to a Vcat
sublayout(::ApplyLayout{typeof(hcat)}, ::Type{<:Tuple{Number,AbstractVector}}) = ApplyLayout{typeof(vcat)}()

arguments(::ApplyLayout{typeof(vcat)}, V::SubArray{<:Any,2,<:Any,<:Tuple{<:Slice,<:Any}}) =
    view.(arguments(parent(V)), Ref(:), Ref(parentindices(V)[2]))
arguments(::ApplyLayout{typeof(hcat)}, V::SubArray{<:Any,2,<:Any,<:Tuple{<:Any,<:Slice}}) =
    view.(arguments(parent(V)), Ref(parentindices(V)[1]), Ref(:))

_vcat_lastinds(sz) = _vcat_cumsum(sz...)
_vcat_firstinds(sz) = (1, (1 .+ most(_vcat_lastinds(sz)))...)

_argsindices(sz) = broadcast(:, _vcat_firstinds(sz), _vcat_lastinds(sz))

_view_vcat(a::Number, kr) = Fill(a,length(kr))
_view_vcat(a::Number, kr, jr) = Fill(a,length(kr), length(jr))
_view_vcat(a, kr...) = view(a, kr...)

function _vcat_sub_arguments(::ApplyLayout{typeof(vcat)}, A, V, kr)
    sz = size.(arguments(A),1)
    skr = intersect.(_argsindices(sz), Ref(kr))
    skr2 = broadcast((a,b) -> a .- b .+ 1, skr, _vcat_firstinds(sz))
    _view_vcat.(arguments(A), skr2)
end

function _vcat_sub_arguments(::ApplyLayout{typeof(vcat)}, A, V, kr, jr)
    sz = size.(arguments(A),1)
    skr = intersect.(_argsindices(sz), Ref(kr))
    skr2 = broadcast((a,b) -> a .- b .+ 1, skr, _vcat_firstinds(sz))
    _view_vcat.(arguments(A), skr2, Ref(jr))
end

_vcat_sub_arguments(LAY::ApplyLayout{typeof(vcat)}, A, V) = _vcat_sub_arguments(LAY, A, V, parentindices(V)...)

_vcat_sub_arguments(::ApplyLayout{typeof(hcat)}, A, V) = arguments(ApplyLayout{typeof(hcat)}(), V)
_vcat_sub_arguments(::PaddedLayout, A, V) = _vcat_sub_arguments(ApplyLayout{typeof(vcat)}(), A, V)

_vcat_sub_arguments(A, V) = _vcat_sub_arguments(MemoryLayout(typeof(A)), A, V)
arguments(::ApplyLayout{typeof(vcat)}, V::SubArray{<:Any,1}) = _vcat_sub_arguments(parent(V), V)

function arguments(L::ApplyLayout{typeof(vcat)}, V::SubArray{<:Any,2})
    A = parent(V)
    args = arguments(L, A)
    kr,jr = parentindices(V)
    sz = size.(args,1)
    skr = intersect.(_argsindices(sz), Ref(kr))
    skr2 = broadcast((a,b) -> a .- b .+ 1, skr, _vcat_firstinds(sz))
    _view_vcat.(args, skr2, Ref(jr))
end

_view_hcat(a::Number, kr, jr) = Fill(a,length(kr),length(jr))
_view_hcat(a, kr, jr) = view(a, kr, jr)

function arguments(L::ApplyLayout{typeof(hcat)}, V::SubArray)
    A = parent(V)
    args = arguments(L, A)
    kr,jr = parentindices(V)
    sz = size.(args,2)
    sjr = intersect.(_argsindices(sz), Ref(jr))
    sjr2 = broadcast((a,b) -> a .- b .+ 1, sjr, _vcat_firstinds(sz))
    _view_hcat.(args, Ref(kr), sjr2)
end

function sub_materialize(::ApplyLayout{typeof(vcat)}, V::AbstractMatrix)
    ret = similar(V)
    n = 0
    _,jr = parentindices(V)
    for a in arguments(V)
        m = size(a,1)
        copyto!(view(ret,n+1:n+m,:), a)
        n += m
    end
    ret
end

sub_materialize(::ApplyLayout{typeof(vcat)}, V::AbstractVector) = ApplyArray(V)

function sub_materialize(::ApplyLayout{typeof(hcat)}, V)
    ret = similar(V)
    n = 0
    kr,_ = parentindices(V)
    for a in arguments(V)
        m = size(a,2)
        copyto!(view(ret,:,n+1:n+m), a)
        n += m
    end
    ret
end

# temporarily allocate. In the future, we add a loop over arguments
materialize!(M::MatMulMatAdd{<:AbstractColumnMajor,<:ApplyLayout{typeof(vcat)}}) =
    materialize!(MulAdd(M.α,M.A,Array(M.B),M.β,M.C))
materialize!(M::MatMulVecAdd{<:AbstractColumnMajor,<:ApplyLayout{typeof(vcat)}}) =
    materialize!(MulAdd(M.α,M.A,Array(M.B),M.β,M.C))

sublayout(::PaddedLayout{L}, ::Type{I}) where {L,I<:Tuple{AbstractUnitRange}} =
    PaddedLayout{typeof(sublayout(L(), I))}()
sublayout(::PaddedLayout{L}, ::Type{I}) where {L,I<:Tuple{AbstractUnitRange,AbstractUnitRange}} =
    PaddedLayout{typeof(sublayout(L(), I))}()

_lazy_getindex(dat, kr...) = view(dat, kr...)
_lazy_getindex(dat::Number, _) = dat

function sub_paddeddata(_, S::SubArray{<:Any,1,<:AbstractVector})
    dat = paddeddata(parent(S))
    (kr,) = parentindices(S)
    kr2 = kr ∩ axes(dat,1)
    _lazy_getindex(dat, kr2)
end

function sub_paddeddata(_, S::SubArray{<:Any,1,<:AbstractMatrix})
    P = parent(S)
    (kr,j) = parentindices(S)
    resizedata!(P, 1, j) # ensure enough rows
    dat = paddeddata(P)
    kr2 = kr ∩ axes(dat,1)
    _lazy_getindex(dat, kr2, j)
end

function sub_paddeddata(_, S::SubArray{<:Any,2})
    dat = paddeddata(parent(S))
    (kr,jr) = parentindices(S)
    kr2 = kr ∩ axes(dat,1)
    _lazy_getindex(dat, kr2, jr)
end

paddeddata(S::SubArray) = sub_paddeddata(MemoryLayout(parent(S)), S)

function _padded_sub_materialize(v::AbstractVector{T}) where T
    dat = paddeddata(v)
    Vcat(dat, Zeros{T}(length(v) - length(dat)))
end

sub_materialize(::PaddedLayout, v::AbstractVector{T}, _) where T =
    _padded_sub_materialize(v)

function sub_materialize(::PaddedLayout, v::AbstractMatrix{T}, _) where T
    dat = paddeddata(v)
    Vcat(dat, Zeros{T}(size(v,1) - size(dat,1), size(dat,2)))
end

## print

_replace_in_print_matrix(A::AbstractArray, k, j, s) = replace_in_print_matrix(A, k, j, s)
_replace_in_print_matrix(_, k, j, s) = s

function layout_replace_in_print_matrix(LAY::ApplyLayout{typeof(vcat)}, f::AbstractVecOrMat, k, j, s)
    κ = k
    for A in arguments(LAY, f)
        n = size(A,1)
        κ ≤ n && return _replace_in_print_matrix(A, κ, j, s)
        κ -= n
    end
    throw(BoundsError(f, (k,j)))
end

function layout_replace_in_print_matrix(::PaddedLayout, f::AbstractVecOrMat, k, j, s)
    data = paddeddata(f)
    k in axes(data,1) ? _replace_in_print_matrix(data, k, j, s) : Base.replace_with_centered_mark(s)
end

# searchsorted

_searchsortedfirst(a, x) = searchsortedfirst(a, x)
_searchsortedfirst(a::Number, x) = 1 + (x > a)
_searchsortedlast(a, x) = searchsortedlast(a, x)
_searchsortedlast(a::Number, x) = 0 + (x ≥ a)

function searchsortedfirst(f::Vcat{<:Any,1}, x)
    n = 0
    for a in arguments(f)
        m = length(a)
        r = _searchsortedfirst(a, x)
        r ≤ m && return n + r
        n += m
    end
    return n+1
end

function searchsortedlast(f::Vcat{<:Any,1}, x)
    args = arguments(f)
    for k in length(args):-1:2
        r = _searchsortedlast(args[k], x)
        r > 0 && return mapreduce(length,+, args[1:k-1]) + r
    end
    return _searchsortedlast(args[1], x)
end

searchsorted(f::Vcat{<:Any,1}, x) = searchsortedfirst(f, x):searchsortedlast(f,x)

# avoid ambiguity in LazyBandedMatrices
copy(M::Mul{<:DiagonalLayout,<:PaddedLayout}) = copy(Lmul(M))


# Triangular columns

sublayout(::TriangularLayout{'U','N', ML}, ::Type{<:Tuple{KR,Integer}}) where {KR,ML} = 
    sublayout(PaddedLayout{ML}(), Tuple{KR})

sublayout(::TriangularLayout{'L','N', ML}, ::Type{<:Tuple{Integer,JR}}) where {JR,ML} = 
    sublayout(PaddedLayout{ML}(), Tuple{JR})

resizedata!(A::UpperTriangular, k::Integer, j::Integer) = resizedata!(parent(A), min(k,j), j)

function sub_paddeddata(::TriangularLayout{'U','N'}, S::SubArray{<:Any,1,<:AbstractMatrix})
    P = parent(S)
    (kr,j) = parentindices(S)
    view(triangulardata(P), kr ∩ (1:j), j)
end
