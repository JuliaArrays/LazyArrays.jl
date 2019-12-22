# Lazy concatenation of AbstractVector's.
# Similar to Iterators.Flatten and some code has been reused from julia/base/iterators.jl


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
@inline axes(f::Vcat{<:Any,2}) = (OneTo(+(map(a -> size(a,1), f.args)...)), OneTo(size(f.args[1],2)))
Base.IndexStyle(::Type{<:Vcat{T,1}}) where T = Base.IndexLinear()
Base.IndexStyle(::Type{<:Vcat{T,2}}) where T = Base.IndexCartesian()


@propagate_inbounds @inline function vcat_getindex(f, k::Integer)
    T = eltype(f)
    κ = k
    for A in f.args
        n = length(A)
        κ ≤ n && return convert(T,A[κ])::T
        κ -= n
    end
    throw(BoundsError(f, k))
end

@propagate_inbounds @inline function vcat_getindex(f, k::Integer, j::Integer)
    T = eltype(f)
    κ = k
    for A in f.args
        n = size(A,1)
        κ ≤ n && return convert(T,A[κ,j])::T
        κ -= n
    end
    throw(BoundsError(f, (k,j)))
end

@propagate_inbounds @inline getindex(f::Vcat{<:Any,1}, k::Integer) = vcat_getindex(f, k)
@propagate_inbounds @inline getindex(f::Vcat{<:Any,2}, k::Integer, j::Integer) = vcat_getindex(f, k, j)
getindex(f::Applied{DefaultArrayApplyStyle,typeof(vcat)}, k::Integer)= vcat_getindex(f, k)
getindex(f::Applied{DefaultArrayApplyStyle,typeof(vcat)}, k::Integer, j::Integer)= vcat_getindex(f, k, j)
getindex(f::Applied{<:Any,typeof(vcat)}, k::Integer)= vcat_getindex(f, k)
getindex(f::Applied{<:Any,typeof(vcat)}, k::Integer, j::Integer)= vcat_getindex(f, k, j)

@propagate_inbounds @inline function setindex!(f::Vcat{T,1}, v, k::Integer) where T
    κ = k
    for A in f.args
        n = length(A)
        κ ≤ n && return setindex!(A, v, κ)
        κ -= n
    end
    throw(BoundsError(f, k))
end

@propagate_inbounds @inline function setindex!(f::Vcat{T,2}, v, k::Integer, j::Integer) where T
    κ = k
    for A in f.args
        n = size(A,1)
        κ ≤ n && return setindex!(A, v, κ, j)
        κ -= n
    end
    throw(BoundsError(f, (k,j)))
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

function hcat_getindex(f, k::Integer, j::Integer)
    T = eltype(f)
    ξ = j
    for A in f.args
        n = size(A,2)
        ξ ≤ n && return T(A[k,ξ])::T
        ξ -= n
    end
    throw(BoundsError(f, (k,j)))
end

getindex(f::Hcat, k::Integer, j::Integer) = hcat_getindex(f, k, j)
getindex(f::Applied{DefaultArrayApplyStyle,typeof(hcat)}, k::Integer, j::Integer)= hcat_getindex(f, k, j)
getindex(f::Applied{<:Any,typeof(hcat)}, k::Integer, j::Integer)= hcat_getindex(f, k, j)

function setindex!(f::Hcat{T}, v, k::Integer, j::Integer) where T
    ξ = j
    for A in f.args
        n = size(A,2)
        ξ ≤ n && return setindex!(A, v, k, ξ)
        ξ -= n
    end
    throw(BoundsError(f, (k,j)))
end


## copyto!
# based on Base/array.jl, Base/abstractarray.jl
copyto!(dest::AbstractArray, V::Vcat) = vcat_copyto!(dest, arguments(V)...)
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

copyto!(dest::AbstractMatrix, H::Hcat) = hcat_copyto!(dest, arguments(H)...)
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
vec(A::Hcat) = Vcat(_vec.(A.args)...)

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
@generated function _vcat_cumsum(x...)
    N = length(x)
    ret = quote
        @nexprs $N d->(c_d = _cumsum(x[d]))
        d_1 = c_1
        @nexprs $(N-1) k->(d_{k+1} = broadcast(+, last(d_k), c_{k+1}))
        @ntuple $N d
    end
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



###
# padded
####

struct PaddedLayout{L} <: MemoryLayout end
applylayout(::Type{typeof(vcat)}, ::A, ::ZerosLayout) where A = PaddedLayout{A}()
cachedlayout(::A, ::ZerosLayout) where A = PaddedLayout{A}()


paddeddata(A::CachedArray) = view(A.data,OneTo.(A.datasize)...)
paddeddata(A::Vcat) = A.args[1]

function ==(A::CachedVector{<:Any,<:Any,<:Zeros}, B::CachedVector{<:Any,<:Any,<:Zeros})
    length(A) == length(B) || return false
    n = max(A.datasize[1], B.datasize[1])
    resizedata!(A,n); resizedata!(B,n)
    view(A.data,OneTo(n)) == view(B.data,OneTo(n))
end

# special copyto! since `similar` of a padded returns a cached
for Typ in (:Number, :AbstractVector)
    @eval function copyto!(dest::CachedVector{T,Vector{T},<:Zeros{T,1}}, src::Vcat{<:Any,1,<:Tuple{<:$Typ,<:Zeros}}) where T
        length(src) ≤ length(dest)  || throw(BoundsError())
        a,_ = src.args
        n = length(a)
        resizedata!(dest, n) # make sure we are padded enough
        copyto!(view(dest.data,OneTo(n)), a)
        dest
    end
end

function copyto!(dest::CachedVector{T,Vector{T},<:Zeros{T,1}}, src::CachedVector{V,Vector{V},<:Zeros{V,1}}) where {T,V}
    length(src) ≤ length(dest)  || throw(BoundsError())
    n = src.datasize[1]
    resizedata!(dest, n)
    copyto!(view(dest.data,OneTo(n)), view(src.data,OneTo(n)))
    dest
end

struct Dot{StyleA,StyleB,ATyp,BTyp}
    A::ATyp
    B::BTyp
end

Dot(A::ATyp,B::BTyp) where {ATyp,BTyp} = Dot{typeof(MemoryLayout(ATyp)), typeof(MemoryLayout(BTyp)), ATyp, BTyp}(A, B)
materialize(d::Dot{<:Any,<:Any,<:AbstractArray,<:AbstractArray}) = Base.invoke(dot, Tuple{AbstractArray,AbstractArray}, d.A, d.B)
function materialize(d::Dot{<:PaddedLayout,<:PaddedLayout,<:AbstractVector{T},<:AbstractVector{V}}) where {T,V}
    a,b = paddeddata(d.A), paddeddata(d.B)
    m = min(length(a), length(b))
    convert(promote_type(T,V), dot(view(a,1:m), view(b,1:m)))
end

dot(a::CachedArray, b::AbstractArray) = materialize(Dot(a,b)) 
dot(a::LazyArray, b::AbstractArray) = materialize(Dot(a,b)) 


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

sublayout(::ApplyLayout{typeof(vcat)}, _) = ApplyLayout{typeof(vcat)}()
sublayout(::ApplyLayout{typeof(hcat)}, _) = ApplyLayout{typeof(hcat)}()
# a row-slice of an Hcat is equivalent to a Vcat
sublayout(::ApplyLayout{typeof(hcat)}, ::Type{<:Tuple{Number,AbstractVector}}) = ApplyLayout{typeof(vcat)}()

arguments(::ApplyLayout{typeof(vcat)}, V::SubArray{<:Any,2,<:Any,<:Tuple{<:Slice,<:Any}}) = 
    view.(arguments(parent(V)), Ref(:), Ref(parentindices(V)[2]))
arguments(::ApplyLayout{typeof(hcat)}, V::SubArray{<:Any,2,<:Any,<:Tuple{<:Any,<:Slice}}) = 
    view.(arguments(parent(V)), Ref(parentindices(V)[1]), Ref(:))

copyto!(dest::AbstractArray{T,N}, src::SubArray{T,N,<:Vcat{T,N}}) where {T,N} = 
    vcat_copyto!(dest, arguments(ApplyLayout{typeof(vcat)}(), src)...)
copyto!(dest::AbstractMatrix{T}, src::SubArray{T,2,<:Hcat{T}}) where T = 
    hcat_copyto!(dest, arguments(ApplyLayout{typeof(hcat)}(), src)...)


_vcat_lastinds(sz) = _vcat_cumsum(sz...)
_vcat_firstinds(sz) = (1, (1 .+ most(_vcat_lastinds(sz)))...)

_argsindices(sz) = broadcast(:, _vcat_firstinds(sz), _vcat_lastinds(sz))

_view_vcat(a::Number, kr) = Fill(a,length(kr))
_view_vcat(a::Number, kr, jr) = Fill(a,length(kr), length(jr))
_view_vcat(a, kr...) = view(a, kr...)

function _vcat_sub_arguments(::ApplyLayout{typeof(vcat)}, A, V)
    kr = parentindices(V)[1]
    sz = size.(arguments(A),1)
    skr = intersect.(_argsindices(sz), Ref(kr))
    skr2 = broadcast((a,b) -> a .- b .+ 1, skr, _vcat_firstinds(sz))
    _view_vcat.(arguments(A), skr2)
end
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
    
_lazy_getindex(dat, kr2) = lazy_getindex(dat, kr2)    
_lazy_getindex(dat::Number, kr2) = dat

function sub_materialize(::PaddedLayout, v::AbstractVector{T}) where T
    A = parent(v)
    dat = paddeddata(A)
    (kr,) = parentindices(v)
    kr2 = kr ∩ axes(dat,1)
    Vcat(_lazy_getindex(dat, kr2), Zeros{T}(length(kr) - length(kr2)))
end

function sub_materialize(::PaddedLayout, v::AbstractMatrix{T}) where T
    A = parent(v)
    dat = paddeddata(A)
    kr,jr = parentindices(v)
    kr2 = kr ∩ axes(dat,1)
    Vcat(lazy_getindex(dat, kr2, jr), Zeros{T}(length(kr) - length(kr2), length(jr)))
end

## print

_replace_in_print_matrix(A::AbstractArray, k, j, s) = replace_in_print_matrix(A, k, j, s)
_replace_in_print_matrix(_, k, j, s) = s

function replace_in_print_matrix(f::Vcat{<:Any,1}, k::Integer, j::Integer, s::AbstractString)
    @assert j == 1
    κ = k
    for A in f.args
        n = length(A)
        κ ≤ n && return _replace_in_print_matrix(A, κ, 1, s)
        κ -= n
    end
    throw(BoundsError(f, k))
end

function replace_in_print_matrix(f::Vcat{<:Any,2}, k::Integer, j::Integer, s::AbstractString)
    κ = k
    for A in f.args
        n = size(A,1)
        κ ≤ n && return _replace_in_print_matrix(A, κ, j, s)
        κ -= n
    end
    throw(BoundsError(f, (k,j)))
end

# searchsorted

# function searchsorted(f::Vcat{<:Any,1}