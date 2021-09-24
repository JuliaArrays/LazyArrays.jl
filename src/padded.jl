
###
# padded
####

struct PaddedLayout{L} <: MemoryLayout end
for op in (:hcat, :vcat)
    @eval begin
        applylayout(::Type{typeof($op)}, ::A, ::ZerosLayout) where A = PaddedLayout{A}()
        applylayout(::Type{typeof($op)}, ::ScalarLayout, ::ScalarLayout, ::ZerosLayout) = PaddedLayout{ApplyLayout{typeof($op)}}()
        applylayout(::Type{typeof($op)}, ::A, ::PaddedLayout) where A = PaddedLayout{ApplyLayout{typeof($op)}}()
        applylayout(::Type{typeof($op)}, ::ScalarLayout, ::ScalarLayout, ::PaddedLayout) = PaddedLayout{ApplyLayout{typeof($op)}}()
    end
end
applylayout(::Type{typeof(hvcat)}, _, ::A, ::ZerosLayout...) where A = PaddedLayout{A}()
cachedlayout(::A, ::ZerosLayout) where A = PaddedLayout{A}()
sublayout(::PaddedLayout{L}, ::Type{I}) where {L,I<:Tuple{AbstractUnitRange}} =
    PaddedLayout{typeof(sublayout(L(), I))}()
sublayout(::PaddedLayout{L}, ::Type{I}) where {L,I<:Tuple{AbstractUnitRange,AbstractUnitRange}} =
    PaddedLayout{typeof(sublayout(L(), I))}()
sublayout(::PaddedLayout{Lay}, sl::Type{<:Tuple{AbstractUnitRange,Integer}}) where Lay =
    PaddedLayout{typeof(sublayout(Lay(), sl))}()
sublayout(::PaddedLayout{Lay}, sl::Type{<:Tuple{Integer,AbstractUnitRange}}) where Lay =
    PaddedLayout{typeof(sublayout(Lay(), sl))}()
transposelayout(::PaddedLayout{L}) where L = PaddedLayout{typeof(transposelayout(L))}()

paddeddata(A::CachedArray{<:Any,N,<:Any,<:Zeros}) where N = cacheddata(A)
_vcat_paddeddata(A, B::Zeros) = A
_vcat_paddeddata(A, B) = Vcat(A, paddeddata(B))
_vcat_paddeddata(A, B, C...) = Vcat(A, _vcat_paddeddata(B, C...))
paddeddata(A::Vcat) = _vcat_paddeddata(A.args...)

_hcat_paddeddata(A, B::Zeros) = A
_hcat_paddeddata(A, B) = Hcat(A, paddeddata(B))
_hcat_paddeddata(A, B, C...) = Hcat(A, _hcat_paddeddata(B, C...))
paddeddata(A::Hcat) = _hcat_paddeddata(A.args...)

paddeddata(A::Transpose) = transpose(paddeddata(parent(A)))
paddeddata(A::Adjoint) = paddeddata(parent(A))'

_hvcat_paddeddata(N, A, B::Zeros...) = A
paddeddata(A::ApplyMatrix{<:Any,typeof(hvcat)}) = _hvcat_paddeddata(A.args...)

const DualOrPaddedLayout{Lay} = Union{PaddedLayout{Lay},DualLayout{PaddedLayout{Lay}}}

function colsupport(lay::DualOrPaddedLayout{Lay}, A, j) where Lay
    P = paddeddata(A)
    MemoryLayout(P) == lay && return colsupport(UnknownLayout, A, j)
    j̃ = j ∩ axes(P,2)
    cs = colsupport(P,j̃)
    isempty(j̃) ? convert(typeof(cs), Base.OneTo(0)) : cs
end
function rowsupport(lay::DualOrPaddedLayout{Lay}, A, k) where Lay
    P = paddeddata(A)
    MemoryLayout(P) == lay && return rowsupport(UnknownLayout, A, j)
    k̃ = k ∩ axes(P,1)
    rs = rowsupport(P,k̃)
    isempty(k̃) ? convert(typeof(rs), Base.OneTo(0)) : rs
end

function _vcat_resizedata!(::DualOrPaddedLayout{Lay}, B, m...) where Lay
    any(iszero,m) || Base.checkbounds(paddeddata(B), m...)
    B
end

resizedata!(B::Vcat, m...) = _vcat_resizedata!(MemoryLayout(B), B, m...)

function ==(A::CachedVector{<:Any,<:Any,<:Zeros}, B::CachedVector{<:Any,<:Any,<:Zeros})
    length(A) == length(B) || return false
    n = max(A.datasize[1], B.datasize[1])
    resizedata!(A,n); resizedata!(B,n)
    view(A.data,OneTo(n)) == view(B.data,OneTo(n))
end

function _copyto!(::PaddedLayout, ::PaddedLayout, dest::AbstractVector, src::AbstractVector)
    length(src) ≤ length(dest)  || throw(BoundsError())
    src_data = paddeddata(src)
    n = length(src_data)
    resizedata!(dest, n) # if resizeable, otherwise this is a no-op
    dest_data = paddeddata(dest)
    copyto!(view(dest_data,OneTo(n)), src_data)
    zero!(view(dest_data,n+1:length(dest_data)))
    dest
end

function _copyto!(::PaddedLayout, ::ZerosLayout, dest::AbstractVector, src::AbstractVector)
    zero!(paddeddata(dest))
    dest
end

function zero!(::PaddedLayout, A)
    zero!(paddeddata(A))
    A
end

ArrayLayouts._norm(::PaddedLayout, A, p) = norm(paddeddata(A), p)


# special case handle broadcasting with padded and cached arrays
function layout_broadcasted(::PaddedLayout, ::PaddedLayout, op, A::AbstractVector, B::AbstractVector)
    a,b = paddeddata(A),paddeddata(B)
    n,m = length(a),length(b)
    dat = if n ≤ m
        [broadcast(op, a, view(b,1:n)); broadcast(op, zero(eltype(A)), @view(b[n+1:end]))]
    else
        [broadcast(op, view(a,1:m), b); broadcast(op, @view(a[m+1:end]), zero(eltype(B)))]
    end
    CachedArray(dat, broadcast(op, Zeros{eltype(A)}(length(A)), Zeros{eltype(B)}(length(B))))
end

function layout_broadcasted(_, ::PaddedLayout, op, A::AbstractVector, B::AbstractVector)
    b = paddeddata(B)
    m = length(b)
    zB = Zeros{eltype(B)}(size(B)...)
    CachedArray(convert(Array,broadcast(op, view(A,1:m), b)), broadcast(op, A, zB))
end

function layout_broadcasted(::PaddedLayout, _, op, A::AbstractVector, B::AbstractVector)
    a = paddeddata(A)
    n = length(a)
    zA = Zeros{eltype(A)}(size(A)...)
    CachedArray(convert(Array,broadcast(op, a, view(B,1:n))), broadcast(op, zA, B))
end

function layout_broadcasted(::PaddedLayout, ::CachedLayout, op, A::AbstractVector, B::AbstractVector)
    a = paddeddata(A)
    n = length(a)
    resizedata!(B,n)
    Bdata = cacheddata(B)
    b = view(Bdata,1:n)
    zA1 = Zeros{eltype(A)}(size(Bdata,1)-n)
    zA = Zeros{eltype(A)}(size(A)...)
    CachedArray([broadcast(op, a, b); broadcast(op, zA1, @view(Bdata[n+1:end]))], broadcast(op, zA, B.array))
end

function layout_broadcasted(::CachedLayout, ::PaddedLayout, op, A::AbstractVector, B::AbstractVector)
    b = paddeddata(B)
    n = length(b)
    resizedata!(A,n)
    Adata = cacheddata(A)
    a = view(Adata,1:n)
    zB1 = Zeros{eltype(B)}(size(Adata,1)-n)
    zB = Zeros{eltype(B)}(size(B)...)
    CachedArray([broadcast(op, a, b); broadcast(op, @view(Adata[n+1:end]), zB1)], broadcast(op, A.array, zB))
end

layout_broadcasted(::ApplyLayout{typeof(vcat)}, lay::PaddedLayout, op, A::AbstractVector, B::AbstractVector) =
    layout_broadcasted(UnknownLayout(), lay, op, A, B)
layout_broadcasted(lay::PaddedLayout, ::ApplyLayout{typeof(vcat)}, op, A::AbstractVector, B::AbstractVector) =
    layout_broadcasted(lay, UnknownLayout(), op, A, B)


# special case for * to preserve Vcat Structure

for op in (:+, :-)
    @eval function layout_broadcasted(::PaddedLayout, ::PaddedLayout, ::typeof($op), A::Vcat{<:Any,1}, B::Vcat{<:Any,1})
        a,b = paddeddata(A),paddeddata(B)
        n,m = length(a),length(b)
        dat = if n > m
            [broadcast($op, view(a,1:m), b); view(a,m+1:n)]
        else # n ≤ m
            [broadcast($op, a, view(b,1:n)); broadcast($op,view(b,n+1:m))]
        end
        Vcat(convert(Array,dat), Zeros{eltype(dat)}(max(length(A),length(B))-length(dat)))
    end
end

function layout_broadcasted(::PaddedLayout, ::PaddedLayout, ::typeof(*), A::Vcat{<:Any,1}, B::Vcat{<:Any,1})
    a,b = paddeddata(A),paddeddata(B)
    n = min(length(a),length(b))
    dat = broadcast(*, view(a,1:n), view(b,1:n))
    Vcat(convert(Array,dat), Zeros{eltype(dat)}(max(length(A),length(B))-n))
end

function layout_broadcasted(_, ::PaddedLayout, ::typeof(*), A::AbstractVector, B::Vcat{<:Any,1})
    b = paddeddata(B)
    m = length(b)
    dat = broadcast(*, view(A,1:m), b)
    Vcat(convert(Array,dat), Zeros{eltype(dat)}(max(length(A),length(B))-m))
end


function layout_broadcasted(::PaddedLayout, _, ::typeof(*), A::Vcat{<:Any,1}, B::AbstractVector)
    a = paddeddata(A)
    n = length(a)
    dat = broadcast(*, a, view(B,1:n))
    Vcat(convert(Array,dat), Zeros{eltype(dat)}(max(length(A),length(B))-n))
end

layout_broadcasted(::PaddedLayout, lay::ApplyLayout{typeof(vcat)}, ::typeof(*), A::Vcat{<:Any,1}, B::AbstractVector) = layout_broadcasted(lay, lay, *, A, B)
layout_broadcasted(lay::ApplyLayout{typeof(vcat)}, ::PaddedLayout, ::typeof(*), A::AbstractVector, B::Vcat{<:Any,1}) = layout_broadcasted(lay, lay, *, A, B)

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


_norm2(::PaddedLayout, a) = norm(paddeddata(a),2)
_norm1(::PaddedLayout, a) = norm(paddeddata(a),1)
_normInf(::PaddedLayout, a) = norm(paddeddata(a),Inf)
_normp(::PaddedLayout, a, p) = norm(paddeddata(a),p)


function copy(D::Dot{layA, layB}) where {layA<:PaddedLayout,layB<:PaddedLayout}
    a = paddeddata(D.A)
    b = paddeddata(D.B)
    T = eltype(D)
    if MemoryLayout(a) isa layA && MemoryLayout(b) isa layB
        return convert(T, dot(Array(a),Array(b)))
    end
    length(a) == length(b) && return convert(T, dot(a,b))
    # following handles scalars
    ((length(a) == 1) || (length(b) == 1)) && return convert(T, a[1] * b[1])
    m = min(length(a), length(b))
    convert(T, dot(view(a, 1:m), view(b, 1:m)))
end

function copy(D::Dot{<:PaddedLayout})
    a = paddeddata(D.A)
    m = length(a)
    v = view(D.B, 1:m)
    if MemoryLayout(a) isa PaddedLayout
        convert(eltype(D), dot(Array(a), v))
    else
        convert(eltype(D), dot(a, v))
    end

end

function copy(D::Dot{<:Any, <:PaddedLayout})
    b = paddeddata(D.B)
    m = length(b)
    v = view(D.A, 1:m)
    if MemoryLayout(b) isa PaddedLayout
        convert(eltype(D), dot(v, Array(b)))
    else
        convert(eltype(D), dot(v, b))
    end
end


_vcat_sub_arguments(::PaddedLayout, A, V) = _vcat_sub_arguments(ApplyLayout{typeof(vcat)}(), A, V)

_lazy_getindex(dat, kr...) = view(dat, kr...)
_lazy_getindex(dat::Number, _...) = dat

function sub_paddeddata(_, S::SubArray{<:Any,1,<:AbstractVector})
    dat = paddeddata(parent(S))
    (kr,) = parentindices(S)
    kr2 = kr ∩ axes(dat,1)
    _lazy_getindex(dat, kr2)
end



function sub_paddeddata(_, S::SubArray{<:Any,1,<:Any,<:Tuple{Integer,Any}})
    P = parent(S)
    (k,jr) = parentindices(S)
    # need to resize in case dat is empty... not clear how to take a vector view of a 0-dimensional matrix in a type-stable way
    resizedata!(P, k, 1); dat = paddeddata(P)
    if k in axes(dat,1)
        _lazy_getindex(dat, k, jr ∩ axes(dat,2))
    else
        _lazy_getindex(dat, first(axes(dat,1)), jr ∩ Base.OneTo(0))
    end
end
function sub_paddeddata(_, S::SubArray{<:Any,1,<:Any,<:Tuple{Any,Integer}})
    P = parent(S)
    (kr,j) = parentindices(S)
    # need to resize in case dat is empty... not clear how to take a vector view of a 0-dimensional matrix in a type-stable way
    resizedata!(P, 1, j); dat = paddeddata(P)
    if j in axes(dat,2)
        _lazy_getindex(dat, kr ∩ axes(dat,1), j)
    else
        _lazy_getindex(dat, kr ∩ Base.OneTo(0), first(axes(dat,2)))
    end
end

function sub_paddeddata(_, S::SubArray{<:Any,2})
    P = parent(S)
    (kr,jr) = parentindices(S)
    dat = paddeddata(P)
    kr2 = kr ∩ axes(dat,1)
    jr2 = jr ∩ axes(dat,2)
    _lazy_getindex(dat, kr2, jr2)
end

paddeddata(S::SubArray) = sub_paddeddata(MemoryLayout(parent(S)), S)

function _padded_sub_materialize(v::AbstractVector{T}) where T
    dat = paddeddata(v)
    Vcat(sub_materialize(dat), Zeros{T}(length(v) - length(dat)))
end

sub_materialize(::PaddedLayout, v::AbstractVector{T}, _) where T =
    _padded_sub_materialize(v)

function sub_materialize(::PaddedLayout, v::AbstractMatrix{T}, _) where T
    dat = paddeddata(v)
    PaddedArray(sub_materialize(dat), size(v)...)
end

function layout_replace_in_print_matrix(::PaddedLayout{Lay}, f::AbstractVecOrMat, k, j, s) where Lay
    # avoid infinite-loop
    f isa SubArray && return Base.replace_in_print_matrix(parent(f), Base.reindex(f.indices, (k,j))..., s)
    data = paddeddata(f)
    MemoryLayout(data) isa PaddedLayout{Lay} && return layout_replace_in_print_matrix(UnknownLayout(), f, k, j, s)
    k in axes(data,1) && j in axes(data,2) && return _replace_in_print_matrix(data, k, j, s)
    Base.replace_with_centered_mark(s)
end

# avoid ambiguity in LazyBandedMatrices
copy(M::Mul{<:DiagonalLayout,<:PaddedLayout}) = copy(Lmul(M))



# Triangular columns

sublayout(::TriangularLayout{'U','N', ML}, INDS::Type{<:Tuple{KR,Integer}}) where {KR,ML} =
    sublayout(PaddedLayout{typeof(sublayout(ML(), INDS))}(), Tuple{KR})

sublayout(::TriangularLayout{'L','N', ML}, INDS::Type{<:Tuple{Integer,JR}}) where {JR,ML} =
    sublayout(PaddedLayout{typeof(sublayout(ML(), INDS))}(), Tuple{JR})


function sub_paddeddata(::TriangularLayout{'U','N'}, S::SubArray{<:Any,1,<:AbstractMatrix,<:Tuple{Any,Integer}})
    P = parent(S)
    (kr,j) = parentindices(S)
    view(triangulardata(P), kr ∩ (1:j), j)
end

function sub_paddeddata(::TriangularLayout{'L','N'}, S::SubArray{<:Any,1,<:AbstractMatrix,<:Tuple{Integer,Any}})
    P = parent(S)
    (k,jr) = parentindices(S)
    view(triangulardata(P), k, jr ∩ (1:k))
end

###
# setindex
###

@inline ndims(A::Applied{<:Any,typeof(setindex)}) = ndims(A.args[1])
@inline eltype(A::Applied{<:Any,typeof(setindex)}) = eltype(A.args[1])
axes(A::ApplyArray{<:Any,N,typeof(setindex)}) where N = axes(A.args[1])

function getindex(A::ApplyVector{T,typeof(setindex)}, k::Integer) where T
    P,v,kr = A.args
    convert(T, k in kr ? v[something(findlast(isequal(k),kr))] : P[k])::T
end

function getindex(A::ApplyMatrix{T,typeof(setindex)}, k::Integer, j::Integer) where T
    P,v,kr,jr = A.args
    convert(T, k in kr && j in jr ? v[something(findlast(isequal(k),kr)),something(findlast(isequal(j),jr))] : P[k,j])::T
end

const PaddedArray{T,N,M} = ApplyArray{T,N,typeof(setindex),<:Tuple{Zeros,M,Vararg{OneTo{Int},N}}}
const PaddedVector{T,M} = PaddedArray{T,1,M}
const PaddedMatrix{T,M} = PaddedArray{T,2,M}

MemoryLayout(::Type{<:PaddedArray{T,N,M}}) where {T,N,M} = PaddedLayout{typeof(MemoryLayout(M))}()
paddeddata(A::PaddedArray) = A.args[2]

PaddedArray(A::AbstractArray{T,N}, n::Vararg{Integer,N}) where {T,N} = ApplyArray{T,N}(setindex, Zeros{T,N}(n...), A, axes(A)...)
PaddedArray(A::T, n::Vararg{Integer,N}) where {T<:Number,N} = ApplyArray{T,N}(setindex, Zeros{T,N}(n...), A, ntuple(_ -> OneTo(1),N)...)