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

@inline Vcat(A::AbstractVector...) = ApplyVector(vcat, A...)
@inline Vcat(A::AbstractVector{T}...) where T = ApplyVector{T}(vcat, A...)
@inline Vcat{T}(A::AbstractVector...) where T = ApplyVector{T}(vcat, A...)

@inline function applied_instantiate(::typeof(vcat), args...)
    iargs = map(instantiate, args)
    if !isempty(iargs)
        m = size(iargs[1],2)
        for k=2:length(iargs)
            size(iargs[k],2) == m || throw(ArgumentError("number of columns of each array must match (got $(map(x->size(x,2), args)))"))
        end
    end
    vcat, iargs
end

@inline applied_eltype(::typeof(vcat)) = Any
@inline applied_eltype(::typeof(vcat), args...) = promote_type(map(eltype, args)...)
@inline applied_ndims(::typeof(vcat), args...) = max(1,maximum(map(ndims,args)))
@inline applied_ndims(::typeof(vcat)) = 1
@inline axes(f::Vcat{<:Any,1,Tuple{}}) = (OneTo(0),)
@inline axes(f::Vcat{<:Any,1}) = tuple(oneto(+(map(length,f.args)...)))
@inline axes(f::Vcat{<:Any,2}) = (oneto(+(map(a -> size(a,1), f.args)...)), axes(f.args[1],2))
@inline size(f::Vcat) = map(length, axes(f))


Base.IndexStyle(::Type{<:Vcat{T,1}}) where T = Base.IndexLinear()

function ==(a::Vcat{T,N}, b::Vcat{T,N}) where {N,T}
    a_args = arguments(vcat, a)
    b_args = arguments(vcat, b)
    if length(a_args) ≠ length(b_args) || any(map(size,a_args) .≠ map(size,b_args))
        return Base.invoke(==, NTuple{2,AbstractArray}, a, b)
    end
    all(a_args .== b_args)
end

@propagate_inbounds @inline vcat_getindex(f, idx...) =
    vcat_getindex_recursive(f, idx, f.args...)

@propagate_inbounds @inline function vcat_getindex_recursive(
        f, idx::Tuple{Integer}, A, args...)
    k, = idx
    T = eltype(f)
    n = length(A)
    k ≤ n && return convert(T, A[k])::T
    vcat_getindex_recursive(f, (k - n, ), args...)
end

@propagate_inbounds @inline function vcat_getindex_recursive(
        f, idx::Tuple{Integer,Integer}, A, args...)
    k, j = idx
    T = eltype(f)
    n = size(A, 1)
    k ≤ n && return convert(T, A[k, j])::T
    vcat_getindex_recursive(f, (k - n, j), args...)
end

@propagate_inbounds @inline function vcat_getindex_recursive(
        f, idx::Tuple{Integer,Union{Colon,AbstractVector}}, A, args...)
    k, j = idx
    T = eltype(f)
    n = size(A, 1)
    k ≤ n && return convert(AbstractVector{T}, A[k, j])
    vcat_getindex_recursive(f, (k - n, j), args...)
end

@inline vcat_getindex_recursive(f, idx) = throw(BoundsError(f, idx))

@propagate_inbounds @inline getindex(f::Vcat{<:Any,1}, k::Integer) = vcat_getindex(f, k)
@propagate_inbounds @inline getindex(f::Vcat{<:Any,2}, k::Integer, j::Integer) = vcat_getindex(f, k, j)
@propagate_inbounds @inline getindex(f::Vcat{<:Any,2}, k::Integer, j::AbstractVector) = vcat_getindex(f, k, j)
@propagate_inbounds @inline getindex(f::Vcat{<:Any,2}, k::Integer, j::Colon) = vcat_getindex(f, k, j)

getindex(f::Applied{DefaultArrayApplyStyle,typeof(vcat)}, k::Integer)= vcat_getindex(f, k)
getindex(f::Applied{DefaultArrayApplyStyle,typeof(vcat)}, k::Integer, j::Integer)= vcat_getindex(f, k, j)
getindex(f::Applied{<:Any,typeof(vcat)}, k::Integer)= vcat_getindex(f, k)
getindex(f::Applied{<:Any,typeof(vcat)}, k::Integer, j::Integer)= vcat_getindex(f, k, j)


# since its mutable we need to make a copy
copy(f::Vcat) = Vcat(map(copy, f.args)...)
map(::typeof(copy), f::Vcat) = Vcat(map.(copy, f.args)...)

@propagate_inbounds @inline function vcat_setindex_recursive!(
        f::Vcat{T,1} where T, v, idx::NTuple{1}, A, args...)
    k, = idx
    n = length(A)
    k ≤ n && return setindex!(A, v, idx...)
    vcat_setindex_recursive!(f, v, (k - n, ), args...)
end

@propagate_inbounds @inline function vcat_setindex_recursive!(
        f::Vcat{T,2} where T, v, idx::NTuple{2}, A, args...)
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

####
# Hcat
####

const Hcat{T,I<:Tuple} = ApplyArray{T,2,typeof(hcat),I}

Hcat(A...) = ApplyArray(hcat, A...)
Hcat() = Hcat{Any}()
Hcat{T}(A...) where T = ApplyArray{T}(hcat, A...)

@inline applied_eltype(::typeof(hcat), args...) = promote_type(map(eltype,args)...)
@inline applied_ndims(::typeof(hcat), args...) = 2
@inline applied_size(::typeof(hcat), args...) = (size(args[1],1), +(map(a -> size(a,2), args)...))
@inline applied_size(::typeof(hcat)) = (0,0)

@inline hcat_getindex(f, k, j::Integer) = hcat_getindex_recursive(f, (k, j), f.args...)

@inline function hcat_getindex_recursive(f, idx::Tuple{Integer,Integer}, A, args...)
    k, j = idx
    T = eltype(f)
    n = size(A, 2)
    j ≤ n && return convert(T, A[k, j])::T
    hcat_getindex_recursive(f, (k, j - n), args...)
end

@inline function hcat_getindex_recursive(f, idx::Tuple{Union{Colon,AbstractVector},Integer}, A, args...)
    kr, j = idx
    T = eltype(f)
    n = size(A, 2)
    j ≤ n && return convert(AbstractVector{T}, A[kr, j])
    hcat_getindex_recursive(f, (kr, j - n), args...)
end

@inline hcat_getindex_recursive(f, idx) = throw(BoundsError(f, idx))

getindex(f::Hcat, k::Integer, j::Integer) = hcat_getindex(f, k, j)
getindex(f::Hcat, k::AbstractVector, j::Integer) = hcat_getindex(f, k, j)
getindex(f::Applied{DefaultArrayApplyStyle,typeof(hcat)}, k::Integer, j::Integer)= hcat_getindex(f, k, j)
getindex(f::Applied{<:Any,typeof(hcat)}, k::Integer, j::Integer)= hcat_getindex(f, k, j)

# since its mutable we need to make a copy
copy(f::Hcat) = Hcat(map(copy, f.args)...)

@inline function hcat_setindex_recursive!(f, v, idx::NTuple{2}, A, args...)
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

####
# Hvcat
####

@inline applied_eltype(::typeof(hvcat), a, b...) = promote_type(map(eltype, b)...)
@inline applied_ndims(::typeof(hvcat), args...) = 2
@inline applied_size(::typeof(hvcat), n::Int, b...) = sum(size.(b[1:n:end],1)),sum(size.(b[1:n],2))

@inline function applied_size(::typeof(hvcat), n::NTuple{N,Int}, b...) where N
    as = tuple(2, (2 .+ cumsum(Base.front(n)))...)
    sum(size.(getindex.(Ref((n, b...)), as),1)),sum(size.(b[1:n[1]],2))
end


@inline hvcat_getindex(f, k, j::Integer) = hvcat_getindex_recursive(f, (k, j), f.args...)

@inline _hvcat_size(A) = size(A)
@inline _hvcat_size(A::Number) = (1,1)
@inline _hvcat_size(A::AbstractVector) = (size(A,1),1)

@inline function hvcat_getindex_recursive(f, (k,j)::Tuple{Integer,Integer}, N::Int, A, args...)
    T = eltype(f)
    m,n = _hvcat_size(A)
    N ≤ 0 && throw(BoundsError(f, (k,j))) # ran out of arrays
    k ≤ m && j ≤ n && return convert(T, A[k, j])::T
    k ≤ m && return hvcat_getindex_recursive(f, (k, j - n), N-1, args...)
    hvcat_getindex_recursive(f, (k - m, j), N, args[N:end]...)
end

@inline function hvcat_getindex_recursive(f, (k,j)::Tuple{Integer,Integer}, N::NTuple{M,Int}, A, args...) where M
    T = eltype(f)
    m,n = _hvcat_size(A)
    k ≤ m && return hvcat_getindex_recursive(f, (k, j), N[1], A, args...)
    hvcat_getindex_recursive(f, (k - m, j), tail(N), args[N[1]:end]...)
end


@inline hvcat_getindex_recursive(f, idx, N) = throw(BoundsError(f, idx))

getindex(f::ApplyMatrix{<:Any,typeof(hvcat)}, k::Integer, j::Integer) = hvcat_getindex(f, k, j)
getindex(f::Applied{<:Any,typeof(hvcat)}, k::Integer, j::Integer)= hvcat_getindex(f, k, j)

#####
# copyto!
####
# based on Base/array.jl, Base/abstractarray.jl
copyto!_layout(_, LAY::ApplyLayout{typeof(vcat)}, dest::AbstractArray{<:Any,N}, V::AbstractArray{<:Any,N}) where N =
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

    i = firstindex(arr)
    for a in arrays
        m = length(a)
        copyto!(view(arr, range(i, length=m)), a)
        i += m
    end
    arr
end

# special case for adjoints of hcat. This is useful for catching fast paths
# for vector case, e.g., _fast_blockbradcast_copyto! in BlockArrays.jl
function vcat_copyto!(dest::AbstractMatrix, arrays::Adjoint{<:Any,<:AbstractVector}...)
    hcat_copyto!(dest', map(adjoint, arrays)...)
    dest
end

copyto!_layout(_, LAY::ApplyLayout{typeof(hcat)}, dest::AbstractMatrix, H::AbstractMatrix) =
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

copyto!_layout(_, lay::ApplyLayout{typeof(hvcat)}, dest::AbstractMatrix, src::AbstractMatrix) = hvcat_copyto!(dest, arguments(lay, src)...)

function hvcat_copyto!(out::AbstractMatrix{T}, nbc::Integer, as...) where T
    # nbc = # of block columns
    n = length(as)
    mod(n,nbc) != 0 &&
        throw(ArgumentError("number of arrays $n is not a multiple of the requested number of block columns $nbc"))
    nbr = div(n,nbc)
    hvcat_copyto!(out, ntuple(i->nbc, nbr), as...)
end

function hvcat_copyto!(a::AbstractMatrix{T}, rows::Tuple{Vararg{Int}}, xs::T...) where T<:Number
    nr = length(rows)
    nc = rows[1]

    size(a) == (nc,nr) || throw(DimensionMismatch())

    if length(a) != length(xs)
        throw(ArgumentError("argument count does not match specified shape (expected $(length(a)), got $(length(xs)))"))
    end
    k = 1
    @inbounds for i=1:nr
        if nc != rows[i]
            throw(ArgumentError("row $(i) has mismatched number of columns (expected $nc, got $(rows[i]))"))
        end
        for j=1:nc
            a[i,j] = xs[k]
            k += 1
        end
    end
    a
end

function hvcat_copyto!(out::AbstractMatrix{T}, rows::Tuple{Vararg{Int}}, as::AbstractVecOrMat...) where T
    nbr = length(rows)  # number of block rows

    nc = 0
    for i=1:rows[1]
        nc += size(as[i],2)
    end

    nr = 0
    a = 1
    for i = 1:nbr
        nr += size(as[a],1)
        a += rows[i]
    end

    size(out) == (nr,nc) || throw(DimensionMismatch())

    a = 1
    r = 1
    for i = 1:nbr
        c = 1
        szi = size(as[a],1)
        for j = 1:rows[i]
            Aj = as[a+j-1]
            szj = size(Aj,2)
            if size(Aj,1) != szi
                throw(ArgumentError("mismatched height in block row $(i) (expected $szi, got $(size(Aj,1)))"))
            end
            if c-1+szj > nc
                throw(ArgumentError("block row $(i) has mismatched number of columns (expected $nc, got $(c-1+szj))"))
            end
            out[r:r-1+szi, c:c-1+szj] = Aj
            c += szj
        end
        if c != nc+1
            throw(ArgumentError("block row $(i) has mismatched number of columns (expected $nc, got $(c-1))"))
        end
        r += szi
        a += rows[i]
    end
    out
end


#####
# adjoint/transpose
#####

for adj in (:adjoint, :transpose)
    @eval begin
        $adj(A::Hcat{T}) where T = Vcat{T}(map($adj,A.args)...)
        $adj(A::Vcat{T,2}) where T = Hcat{T}(map($adj,A.args)...)
    end
end

_vec(a) = a
_vec(a::AbstractArray) = vec(a)
_vec(a::Adjoint{<:Number,<:AbstractVector}) = _vec(parent(a))
vec(A::Hcat) = Vcat(map(_vec,A.args)...)

copy(f::Adjoint{<:Any,<:Union{Vcat,Hcat}}) = copy(parent(f))'
copy(f::Transpose{<:Any,<:Union{Vcat,Hcat}}) = transpose(copy(parent(f)))

_permutedims(a) = a
_permutedims(a::AbstractArray) = permutedims(a)

permutedims(A::Hcat{T}) where T = Vcat{T}(map(_permutedims,A.args)...)
permutedims(A::Vcat{T}) where T = Hcat{T}(map(_permutedims,A.args)...)

transposelayout(::ApplyLayout{typeof(vcat)}) = ApplyLayout{typeof(hcat)}()
transposelayout(::ApplyLayout{typeof(hcat)}) = ApplyLayout{typeof(vcat)}()

arguments(::ApplyLayout{typeof(vcat)}, A::Adjoint) = map(adjoint, arguments(ApplyLayout{typeof(hcat)}(), parent(A)))
arguments(::ApplyLayout{typeof(hcat)}, A::Adjoint) = map(adjoint, arguments(ApplyLayout{typeof(vcat)}(), parent(A)))
arguments(::ApplyLayout{typeof(vcat)}, A::Transpose) = map(transpose, arguments(ApplyLayout{typeof(hcat)}(), parent(A)))
arguments(::ApplyLayout{typeof(hcat)}, A::Transpose) = map(transpose, arguments(ApplyLayout{typeof(vcat)}(), parent(A)))

function arguments(::ApplyLayout{typeof(vcat)}, C::CachedVector)
    data = cacheddata(C)
    Vcat(data, C.array[length(data)+1:end])
end


#####
# broadcasting
#
# We want broadcasting for numbers with concatenations to pass through
# to take advantage of special implementations of the sub-components
######

BroadcastStyle(::Type{<:Vcat{<:Any,N}}) where N = LazyArrayStyle{N}()
BroadcastStyle(::Type{<:Hcat{<:Any}}) = LazyArrayStyle{2}()

# This is if we broadcast a function on a mixed concat f.([1; [2,3]])
# such that f returns a vector, e.g., f(1) == [1,2], we don't want
# to have the concat return [f(1); [f(2),f(3)]] but rather [[f(1)]; [f(2),f(3)]]

_flatten_nums(args::Tuple{}, bc::Tuple{}) = ()
_flatten_nums(args::Tuple, bc::Tuple) = (bc[1], _flatten_nums(tail(args), tail(bc))...)
_flatten_nums(args::Tuple{Number, Vararg{Any}}, bc::Tuple{AbstractArray, Vararg{Any}}) = (Fill(bc[1],1), _flatten_nums(tail(args), tail(bc))...)

broadcasted(::LazyArrayStyle, op, A::Vcat) = Vcat(_flatten_nums(A.args, broadcast(x -> broadcast(op, x), A.args))...)
broadcasted(::LazyArrayStyle, op, A::Transpose{<:Any,<:Vcat}) = transpose(broadcast(op, parent(A)))
broadcasted(::LazyArrayStyle, op, A::Adjoint{<:Real,<:Vcat}) = broadcast(op, parent(A))'


for Cat in (:Vcat, :Hcat)
     @eval begin
        broadcasted(::LazyArrayStyle, op, A::$Cat, c::Number) = $Cat(_flatten_nums(A.args, broadcast((x,y) -> broadcast(op, x, y), A.args, c))...)
        broadcasted(::LazyArrayStyle, op, c::Number, A::$Cat) = $Cat(_flatten_nums(A.args, broadcast((x,y) -> broadcast(op, x, y), c, A.args))...)
        broadcasted(::LazyArrayStyle, op, A::$Cat, c::Ref) = $Cat(_flatten_nums(A.args, broadcast((x,y) -> broadcast(op, x, Ref(y)), A.args, c))...)
        broadcasted(::LazyArrayStyle, op, c::Ref, A::$Cat) = $Cat(_flatten_nums(A.args, broadcast((x,y) -> broadcast(op, Ref(x), y), c, A.args))...)
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

# let it stay lazy
layout_broadcasted(::ApplyLayout{typeof(vcat)}, ::AbstractLazyLayout, op, A::AbstractVector, B::AbstractVector) =
    Broadcasted{LazyArrayStyle{1}}(op, (A, B))
layout_broadcasted(::AbstractLazyLayout, ::ApplyLayout{typeof(vcat)}, op, A::AbstractVector, B::AbstractVector) =
    Broadcasted{LazyArrayStyle{1}}(op, (A, B))

layout_broadcasted(::ApplyLayout{typeof(vcat)}, lay::CachedLayout, op, A::AbstractVector, B::AbstractVector) = layout_broadcasted(UnknownLayout(), lay, op, A, B)
layout_broadcasted(lay::CachedLayout, ::ApplyLayout{typeof(vcat)}, op, A::AbstractVector, B::AbstractVector) = layout_broadcasted(lay, UnknownLayout(), op, A, B)

function layout_broadcasted(::ApplyLayout{typeof(vcat)}, _, op, A::AbstractVector, B::AbstractVector)
    kr = _vcat_axes(map(axes,A.args)...)  # determine how to break up B
    B_arrays = _vcat_getindex_eval(B,kr...)    # evaluate B at same chunks as A
    ApplyVector(vcat, broadcast((a,b) -> broadcast(op,a,b), A.args, B_arrays)...)
end

function layout_broadcasted(_, ::ApplyLayout{typeof(vcat)}, op, A::AbstractVector, B::AbstractVector)
    kr = _vcat_axes(axes.(B.args)...)
    A_arrays = _vcat_getindex_eval(A,kr...)
    Vcat(broadcast((a,b) -> broadcast(op,a,b), A_arrays, B.args)...)
end



######
# Special Vcat broadcasts
#
# We use Vcat for infinite padded vectors, so we need to special case
# two arrays. This may be generalisable in the future
######

layout_broadcasted(lay::ApplyLayout{typeof(vcat)}, ::ApplyLayout{typeof(vcat)}, op, A::AbstractVector, B::AbstractVector) =
    _vcat_layout_broadcasted(arguments(lay, A), arguments(lay, B), op, A, B)

_vcat_layout_broadcasted(Aargs, Bargs, op, A, B) = Broadcasted{LazyArrayStyle{1}}(op, (A,B))

function _vcat_layout_broadcasted((Ahead,Atail)::Tuple{AbstractVector,Any}, (Bhead,Btail)::Tuple{AbstractVector,Any}, op, A, B)
    T = Broadcast.combine_eltypes(op, (eltype(A), eltype(B)))

    if length(Ahead) ≥ length(Bhead)
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

function _vcat_layout_broadcasted((Ahead,Atail)::Tuple{AbstractVector,Any}, (Bhead,Btail)::Tuple{Number,Any}, op, A, B)
    T = Broadcast.combine_eltypes(op, (eltype(A), eltype(B)))
    M = length(Ahead)
    Chead = Vector{T}(undef,max(1,M))
    Chead[1] = op(A[1], Bhead)
    view(Chead,2:M) .= op.(view(Ahead,2:M),Btail[1:M-1])
    Ctail = op.(Atail, Btail[M:end])
    Vcat(Chead, Ctail)
end

function _vcat_layout_broadcasted((Ahead,Atail)::Tuple{Number,Any}, (Bhead,Btail)::Tuple{AbstractVector,Any}, op, A, B)
    T = Broadcast.combine_eltypes(op, (eltype(A), eltype(B)))
    M = length(Bhead)
    Chead = Vector{T}(undef,max(1,M))
    Chead[1] = op(Ahead, B[1])
    view(Chead,2:M) .= op.(Atail[1:M-1],view(Bhead,2:M))
    Ctail = op.(Atail[M:end],Btail)
    Vcat(Chead, Ctail)
end

_vcat_layout_broadcasted((Ahead,Atail)::Tuple{Number,Any}, (Bhead,Btail)::Tuple{Number,Any}, op, A, B) = Vcat(op.(Ahead,Bhead), op.(Atail,Btail))


broadcasted(::LazyArrayStyle, op, a::Vcat{<:Any,N}, b::AbstractArray{<:Any,N}) where N = layout_broadcasted(op, a, b)
broadcasted(::LazyArrayStyle, op, a::AbstractArray{<:Any,N}, b::Vcat{<:Any,N}) where N = layout_broadcasted(op, a, b)
broadcasted(::LazyArrayStyle{1}, op, a::Vcat{<:Any,1}, b::Zeros{<:Any,1}) = broadcast(DefaultArrayStyle{1}(), op, a, b)
broadcasted(::LazyArrayStyle{1}, op, a::Zeros{<:Any,1}, b::Vcat{<:Any,1}) = broadcast(DefaultArrayStyle{1}(), op, a, b)
broadcasted(::LazyArrayStyle{1}, ::typeof(\), a::Vcat{<:Any,1}, b::Zeros{<:Any,1}) = broadcast(DefaultArrayStyle{1}(), \, a, b)
broadcasted(::LazyArrayStyle{1}, ::typeof(/), a::Zeros{<:Any,1}, b::Vcat{<:Any,1}) = broadcast(DefaultArrayStyle{1}(), /, a, b)
broadcasted(::LazyArrayStyle{1}, ::typeof(*), a::Vcat{<:Any,1}, b::Zeros{<:Any,1}) = broadcast(DefaultArrayStyle{1}(), *, a, b)
broadcasted(::LazyArrayStyle{1}, ::typeof(*), a::Zeros{<:Any,1}, b::Vcat{<:Any,1}) = broadcast(DefaultArrayStyle{1}(), *, a, b)


# Cannot broadcast Vcat's in a lazy way so stick to BroadcastArray
broadcasted(::LazyArrayStyle, op, A::Vcat, B::Vcat) = layout_broadcasted(op, A, B)

# ambiguities
broadcasted(::LazyArrayStyle, op, A::Vcat{<:Any,1}, B::CachedVector) = layout_broadcasted(op, A, B)
broadcasted(::LazyArrayStyle, op, A::CachedVector, B::Vcat{<:Any,1}) = layout_broadcasted(op, A, B)



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
@inline _cumsum_last(x::AbstractVector{T}) where T = isempty(x) ? zero(T) : last(x)
@inline _cumsum_last(x) = last(x)

_tuple_cumsum() = ()
_tuple_cumsum(a) = (a,)
_tuple_cumsum(a, b...) = (a, broadcast(+,a,_tuple_cumsum(b...))...)
function _vcat_cumsum(x...)
    cs = map(_cumsum,x)
    cslasts = tuple(0,_tuple_cumsum(map(_cumsum_last,Base.front(cs))...)...)
    map((a,b) -> broadcast(+,a,b), cslasts, cs)
end

@inline cumsum(V::Vcat{<:Any,1}) = ApplyVector(vcat,_vcat_cumsum(V.args...)...)
# For simplicity we just use accumulate
@inline accumulate(::typeof(+), V::Vcat{<:Any,1}) = cumsum(V)

###
# cumsum(Vcat(::Number, ::Fill))
# special override. Used with BlockArrays
###

@inline function cumsum(v::Vcat{T,1,<:Tuple{Number,AbstractFill}}) where T
    V = promote_op(add_sum, T, T)
    a,b = v.args
    FillArrays.steprangelen(convert(V, a), getindex_value(b), length(b)+1)
end

@inline function cumsum(v::Vcat{T,1,<:Tuple{Number,Zeros}}) where T
    a,b = v.args
    V = promote_op(add_sum, T, T)
    Fill(convert(V,a), length(b)+1)
end

@inline function cumsum(v::Vcat{T,1,<:Tuple{Number,Ones}}) where T
    a,b = v.args
    V = promote_op(add_sum, T, T)
    convert(V,a) .+ range(zero(V); length=length(b)+1)
end

for op in (:+, :-)
    @eval @inline function accumulate(::typeof($op), v::Vcat{T,1,<:Tuple{Number,AbstractFill}}) where T
        V = promote_op(add_sum, T, T)
        a,b = v.args
        FillArrays.steprangelen(convert(V, a), $op(getindex_value(b)), length(b)+1)
    end
end

@inline function accumulate(::typeof(+), v::Vcat{T,1,<:Tuple{Number,Zeros}}) where T
    a,b = v.args
    V = promote_op(+, T, T)
    Fill(convert(V,a), length(b)+1)
end

@inline function accumulate(::typeof(+), v::Vcat{T,1,<:Tuple{Number,Ones}}) where T
    a,b = v.args
    V = promote_op(+, T, T)
    convert(V,a) .+ range(zero(V); length=length(b)+1)
end



_vcat_diff(x::Number) = ()
_vcat_diff(x) = (diff(x),)

_vcat_diff(a::Number, b, c...) = (first(b)-a, _vcat_diff(b,c...)...)
_vcat_diff(a, b, c...) = (diff(a), first(b)-last(a), _vcat_diff(b,c...)...)
@inline diff(V::Vcat{T,1}) where T = ApplyVector{T}(vcat,_vcat_diff(V.args...)...)

####
# maximum/minimum
####

sum(v::Vcat{<:Any,1}) = sum(map(sum,v.args))

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

copy(M::Mul{ApplyLayout{typeof(vcat)},<:AbstractLazyLayout}) = vcat((arguments(vcat, M.A) .* Ref(M.B))...)

_all_prods(a::Tuple{}, ::Tuple) = ()
_all_prods(a::Tuple, b::Tuple) = tuple((Ref(first(a)) .* b)..., _all_prods(tail(a), b)...)
function copy(M::Mul{ApplyLayout{typeof(vcat)},ApplyLayout{typeof(hcat)}})
    b = arguments(hcat,M.B)
    ApplyArray(hvcat, length(b), _all_prods(arguments(vcat, M.A), b)...)
end


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

function colsupport(lay::ApplyLayout{typeof(vcat)}, M::AbstractArray, j)
    args = arguments(lay, M)
    first(colsupport(first(args),j)):(size(Vcat(Base.front(args)...),1)+last(colsupport(last(args),j)))
end

function rowsupport(lay::ApplyLayout{typeof(vcat)}, V::AbstractArray, k::Integer)
    ξ = k
    for A in arguments(lay, V)
        n = size(A,1)
        ξ ≤ n && return rowsupport(A, ξ)
        ξ -= n
    end
    return 1:0
end

function colsupport(lay::ApplyLayout{typeof(hcat)}, H::AbstractArray, j::Integer)
    ξ = j
    for A in arguments(lay,H)
        n = size(A,2)
        ξ ≤ n && return colsupport(A, ξ)
        ξ -= n
    end
    return 1:0
end

function rowsupport(lay::ApplyLayout{typeof(hcat)}, M::AbstractArray, k)
    args = arguments(lay, M)
    first(rowsupport(first(args),k)):(size(Hcat(Base.front(args)...),2)+last(rowsupport(last(args),k)))
end

include("padded.jl")



###
# norm
###

for Cat in (:Vcat, :Hcat)
    for (op,p) in ((:norm1,1), (:norm2,2), (:normInf,Inf))
        @eval $op(a::$Cat) = $op(norm.(a.args,$p))
    end
    @eval normp(a::$Cat, p) = norm(norm.(a.args, p), p)
end

###
# subarrays
###

sublayout(::ApplyLayout{typeof(vcat)}, ::Type{<:Tuple{Vararg{Union{AbstractRange{Int},Int}}}}) = ApplyLayout{typeof(vcat)}()
sublayout(::ApplyLayout{typeof(hcat)}, ::Type{<:Tuple{Vararg{Union{AbstractRange{Int},Int}}}}) = ApplyLayout{typeof(hcat)}()
# a row-slice of an Hcat is equivalent to a Vcat
sublayout(::ApplyLayout{typeof(hcat)}, ::Type{<:Tuple{Int,AbstractRange{Int}}}) = ApplyLayout{typeof(vcat)}()

_vcat_lastinds(sz) = _vcat_cumsum(sz...)
_vcat_firstinds(sz) = (1, (1 .+ Base.front(_vcat_lastinds(sz)))...)

_argsindices(sz) = broadcast(:, _vcat_firstinds(sz), _vcat_lastinds(sz))

_view_vcat(a::Number, kr) = Fill(a,length(kr))
_view_vcat(a::Number, kr, jr) = Fill(a,length(kr), length(jr))
_view_vcat(a, kr...) = _viewifmutable(a, kr...)

_reverse_if_neg_step(args, kr::AbstractUnitRange) = args
_reverse_if_neg_step(args, kr::AbstractRange) = step(kr) ≥ 0 ? args : reverse(args)

function _vcat_sub_arguments(lay::ApplyLayout{typeof(vcat)}, A, V, kr)
    sz = size.(arguments(lay, A),1)
    skr = intersect.(_argsindices(sz), Ref(kr))
    skr2 = broadcast((a,b) -> a .- b .+ 1, skr, _vcat_firstinds(sz))
    _reverse_if_neg_step(map(_view_vcat, arguments(lay, A), skr2), kr)
end

function _vcat_sub_arguments(::ApplyLayout{typeof(vcat)}, A, V, kr, jr)
    sz = size.(arguments(A),1)
    skr = intersect.(_argsindices(sz), Ref(kr))
    skr2 = broadcast((a,b) -> a .- b .+ 1, skr, _vcat_firstinds(sz))
    _reverse_if_neg_step(_view_vcat.(arguments(A), skr2, Ref(jr)), kr)
end

_vcat_sub_arguments(LAY::ApplyLayout{typeof(vcat)}, A, V) = _vcat_sub_arguments(LAY, A, V, parentindices(V)...)


function _vcat_sub_arguments(L::ApplyLayout{typeof(hcat)}, A, V)
    A = parent(V)
    args = arguments(L, A)
    k,jr = parentindices(V)
    sz = size.(args,2)
    sjr = intersect.(_argsindices(sz), Ref(jr))
    sjr2 = broadcast((a,b) -> a .- b .+ 1, sjr, _vcat_firstinds(sz))
    _view_hcat.(_reverse_if_neg_step(args, jr), k, sjr2)
end

_vcat_sub_arguments(::DualLayout{ML}, A, V) where ML = _vcat_sub_arguments(ML(), A, V)
_vcat_sub_arguments(A, V) = _vcat_sub_arguments(MemoryLayout(typeof(A)), A, V)
arguments(::ApplyLayout{typeof(vcat)}, V::SubArray{<:Any,1}) = _vcat_sub_arguments(parent(V), V)



function arguments(L::ApplyLayout{typeof(vcat)}, V::SubArray{<:Any,2})
    A = parent(V)
    args = arguments(L, A)
    kr,jr = parentindices(V)
    sz = size.(args,1)
    skr = intersect.(_argsindices(sz), Ref(kr))
    skr2 = broadcast((a,b) -> a .- b .+ 1, skr, _vcat_firstinds(sz))
    _view_vcat.(_reverse_if_neg_step(args, kr), skr2, Ref(jr))
end

@inline _view_hcat(a::Number, kr, jr) = Fill(a,length(kr),length(jr))
@inline _view_hcat(a::Number, kr::Number, jr) = Fill(a,length(jr))
@inline _view_hcat(a, kr, jr) = _viewifmutable(a, kr, jr)
@inline _view_hcat(a::AbstractVector, kr, jr::Colon) = _viewifmutable(a, kr)

# equivalent to broadcast but written to maintain type stability
__view_hcat(::Tuple{}, _, ::Tuple{}) = ()
__view_hcat(::Tuple{}, _, ::Colon) = ()
@inline __view_hcat(args::Tuple, kr, jrs::Tuple) = (_view_hcat(args[1], kr, jrs[1]), __view_hcat(tail(args), kr, tail(jrs))...)
@inline __view_hcat(args::Tuple, kr, ::Colon) = (_view_hcat(args[1], kr, :), __view_hcat(tail(args), kr, :)...)

function arguments(L::ApplyLayout{typeof(hcat)}, V::SubArray)
    A = parent(V)
    args = arguments(L, A)
    kr,jr = parentindices(V)
    sz = size.(args,2)
    sjr = intersect.(_argsindices(sz), Ref(jr))
    sjr2 = broadcast((a,b) -> a .- b .+ 1, sjr, _vcat_firstinds(sz))
    __view_hcat(args, kr, sjr2)
end

arguments(::ApplyLayout{typeof(vcat)}, V::SubArray{<:Any,2,<:Any,<:Tuple{<:Slice,<:Any}}) =
    _viewifmutable.(arguments(parent(V)), Ref(:), Ref(parentindices(V)[2]))
arguments(::ApplyLayout{typeof(hcat)}, V::SubArray{<:Any,2,<:Any,<:Tuple{<:Any,<:Slice}}) =
    __view_hcat(arguments(parent(V)), parentindices(V)[1], :)

function sub_materialize(lay::ApplyLayout{typeof(vcat)}, V::AbstractMatrix, _)
    ret = similar(V)
    n = 0
    _,jr = parentindices(V)
    for a in arguments(lay, V)
        m = size(a,1)
        copyto!(view(ret,n+1:n+m,:), a)
        n += m
    end
    ret
end

sub_materialize(::ApplyLayout{typeof(vcat)}, V::AbstractVector, _) = ApplyVector(V)

function sub_materialize(::ApplyLayout{typeof(hcat)}, V, _)
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

## print

_replace_in_print_matrix(A::AbstractArray, k, j, s) = replace_in_print_matrix(A, k, j, s)
_replace_in_print_matrix(_, k, j, s) = s

function layout_replace_in_print_matrix(LAY::ApplyLayout{typeof(vcat)}, f::AbstractVecOrMat, k, j, s)
    κ = k
    for A in arguments(LAY, f)
        n = size(A,1)
        κ ≤ n && return _replace_in_print_matrix(A, κ, j, s)
        κ -= n
    end
    throw(BoundsError(f, (k,j)))
end

# searchsorted

_searchsortedfirst(a, x) = searchsortedfirst(a, x)
_searchsortedfirst(a::Number, x) = 1 + (x > a)
_searchsortedlast(a, x) = searchsortedlast(a, x)
_searchsortedlast(a::Number, x) = 0 + (x ≥ a)

searchsortedfirst(f::Vcat{<:Any,1}, x) =
    searchsortedfirst_recursive(0, x, arguments(vcat, f)...)

searchsortedlast(f::Vcat{<:Any,1}, x) =
    searchsortedlast_recursive(length(f), x, reverse(arguments(vcat, f))...)

@inline searchsortedfirst_recursive(n, x) = n + 1

@inline function searchsortedfirst_recursive(n, x, a, args...)
    m = length(a)
    r = _searchsortedfirst(a, x)
    r ≤ m && return n + r
    return searchsortedfirst_recursive(n + m, x, args...)
end

@inline searchsortedlast_recursive(n, x) = n

@inline function searchsortedlast_recursive(n, x, a, args...)
    n -= length(a)
    r = _searchsortedlast(a, x)
    r > 0 && return n + r
    return searchsortedlast_recursive(n, x, args...)
end

searchsorted(f::Vcat{<:Any,1}, x) = searchsortedfirst(f, x):searchsortedlast(f,x)

###
# vec
###

@inline applied_eltype(::typeof(vec), a) = eltype(a)
@inline applied_axes(::typeof(vec), a) = (oneto(length(a)),)
@inline applied_ndims(::typeof(vec), a) = 1
@inline applied_size(::typeof(vec), a) = (length(a),)
