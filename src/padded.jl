
###
# padded
####

abstract type AbstractPaddedLayout{L} <: MemoryLayout end


"""
    PaddedColumns{L}()

represents a vector or matrix with layout `L()` whose columns have been padded with zeros
below, i.e., a lazy version of `[A; Zeros(...)]`.
"""
struct PaddedColumns{L} <: AbstractPaddedLayout{L} end
"""
    PaddedRows{L}()

represents a matrix with layout `L()` whose rows have been padded with zeros
below, i.e., a lazy version of `[A Zeros(...)]`.
"""
struct PaddedRows{L} <: AbstractPaddedLayout{L} end

"""
    PaddedLayout{L}()

represents a matrix whose rows and columns have been padded.
"""
struct PaddedLayout{L} <: AbstractPaddedLayout{L} end

applylayout(::Type{typeof(vcat)}, ::A, ::ZerosLayout) where A = PaddedColumns{A}()
applylayout(::Type{typeof(vcat)}, ::ScalarLayout, ::ScalarLayout, ::ZerosLayout) = PaddedColumns{ApplyLayout{typeof(vcat)}}()
applylayout(::Type{typeof(vcat)}, ::A, ::PaddedColumns) where A = PaddedColumns{ApplyLayout{typeof(vcat)}}()
applylayout(::Type{typeof(vcat)}, ::ScalarLayout, ::ScalarLayout, ::PaddedColumns) = PaddedColumns{ApplyLayout{typeof(vcat)}}()

applylayout(::Type{typeof(hcat)}, ::A, ::ZerosLayout) where A = PaddedRows{A}()
applylayout(::Type{typeof(hcat)}, ::ScalarLayout, ::ScalarLayout, ::ZerosLayout) = PaddedRows{ApplyLayout{typeof(vcat)}}()
applylayout(::Type{typeof(hcat)}, ::A, ::PaddedRows) where A = PaddedRows{ApplyLayout{typeof(vcat)}}()
applylayout(::Type{typeof(hcat)}, ::A, ::PaddedLayout) where A = PaddedLayout{ApplyLayout{typeof(vcat)}}()
applylayout(::Type{typeof(hcat)}, ::ScalarLayout, ::ScalarLayout, ::PaddedRows) = PaddedRows{ApplyLayout{typeof(vcat)}}()
applylayout(::Type{typeof(hcat)}, ::DualLayout, ::DualLayout{<:PaddedRows}) = DualLayout{PaddedRows{ApplyLayout{typeof(hcat)}}}()


applylayout(::Type{typeof(hvcat)}, _, ::A, ::ZerosLayout...) where A = PaddedLayout{A}()
cachedlayout(::A, ::ZerosLayout) where A = PaddedLayout{A}()
MemoryLayout(C::Type{CachedVector{T,DAT,ARR}}) where {T,DAT,ARR<:Zeros} = PaddedColumns{typeof(MemoryLayout(DAT))}()


similarpadded(::PaddedColumns, ::L) where L = PaddedColumns{L}()
similarpadded(::PaddedRows, ::L) where L = PaddedRows{L}()
similarpadded(::PaddedLayout, ::L) where L = PaddedLayout{L}()


sublayout(::PaddedColumns{L}, ::Type{I}) where {L,I<:Tuple{AbstractUnitRange}} =
    PaddedColumns{typeof(sublayout(L(), I))}()
sublayout(pad::AbstractPaddedLayout{L}, ::Type{I}) where {L,I<:Tuple{AbstractUnitRange,AbstractUnitRange}} =
    similarpadded(pad, sublayout(L(), I))
sublayout(::Union{PaddedLayout{Lay}, PaddedColumns{Lay}}, sl::Type{<:Tuple{AbstractUnitRange,Integer}}) where Lay =
    PaddedColumns{typeof(sublayout(Lay(), sl))}()
sublayout(::Union{PaddedLayout{Lay}, PaddedRows{Lay}}, sl::Type{<:Tuple{Integer,AbstractUnitRange}}) where Lay =
    PaddedColumns{typeof(sublayout(Lay(), sl))}()
transposelayout(::PaddedLayout{L}) where L = PaddedLayout{typeof(transposelayout(L))}()
transposelayout(::PaddedRows{L}) where L = PaddedColumns{typeof(transposelayout(L))}()
transposelayout(::PaddedColumns{L}) where L = PaddedRows{typeof(transposelayout(L))}()

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




function colsupport(lay::PaddedColumns{Lay}, A, j) where Lay
    P = paddeddata(A)
    MemoryLayout(P) == lay && return colsupport(UnknownLayout, P, j)
    colsupport(P,j)
end

function colsupport(lay::Union{DualLayout{PaddedRows{Lay}}, PaddedRows{Lay}, PaddedLayout{Lay}}, A, j) where Lay
    P = paddeddata(A)
    MemoryLayout(P) == lay && return colsupport(UnknownLayout, P, j)
    j̃ = j ∩ axes(P,2)
    cs = colsupport(P,j̃)
    isempty(j̃) ? convert(typeof(cs), Base.OneTo(0)) : cs
end

function rowsupport(lay::Union{DualLayout{PaddedRows{Lay}}, PaddedRows{Lay}}, A, k) where Lay
    P = paddeddata(A)
    MemoryLayout(P) == lay && return rowsupport(UnknownLayout, P, k)
    rowsupport(P,k)
end

function rowsupport(lay::Union{PaddedColumns{Lay}, PaddedLayout{Lay}}, A, k) where Lay
    P = paddeddata(A)
    MemoryLayout(P) == lay && return rowsupport(UnknownLayout, P, k)
    k̃ = k ∩ axes(P,1)
    rs = rowsupport(P,k̃)
    isempty(k̃) ? convert(typeof(rs), Base.OneTo(0)) : rs
end

function _vcat_resizedata!(::Union{AbstractPaddedLayout, DualLayout{<:PaddedRows}}, B, m...)
    any(iszero,m) || Base.checkbounds(paddeddata(B), m...)
    B
end

_vcat_resizedata!(_, B, m...) = B # by default we can't resize

resizedata!(B::Vcat, m...) = _vcat_resizedata!(MemoryLayout(B), B, m...)

function ==(A::CachedVector{<:Any,<:Any,<:Zeros}, B::CachedVector{<:Any,<:Any,<:Zeros})
    length(A) == length(B) || return false
    n = max(A.datasize[1], B.datasize[1])
    resizedata!(A,n); resizedata!(B,n)
    view(A.data,OneTo(n)) == view(B.data,OneTo(n))
end

function ==(A::CachedArray{<:Any,<:Any,<:Any,<:Zeros}, B::CachedArray{<:Any,<:Any,<:Any,<:Zeros})
    size(A) == size(B) || return false
    m = max(A.datasize[1], B.datasize[1])
    n = max(A.datasize[2], B.datasize[2])
    resizedata!(A, m, n); resizedata!(B, m, n)
    view(A.data, OneTo(m), OneTo(n)) == view(B.data, OneTo(m), OneTo(n))
end

function copyto!_layout(::PaddedColumns, ::PaddedColumns, dest::AbstractVector, src::AbstractVector)
    length(src) ≤ length(dest)  || throw(BoundsError())
    src_data = paddeddata(src)
    n = length(src_data)
    resizedata!(dest, n) # if resizeable, otherwise this is a no-op
    dest_data = paddeddata(dest)
    copyto!(view(dest_data,OneTo(n)), src_data)
    zero!(view(dest_data,n+1:length(dest_data)))
    dest
end


function copyto!_layout(::AbstractPaddedLayout, ::AbstractPaddedLayout, dest::AbstractMatrix, src::AbstractMatrix)
    (size(src,1) ≤ size(dest,1) && size(src,2) ≤ size(dest,2))  || throw(BoundsError())
    src_data = paddeddata(src)
    m,n = size(src_data)
    resizedata!(dest, m, n) # if resizeable, otherwise this is a no-op
    dest_data = paddeddata(dest)
    copyto!(view(dest_data,OneTo(m),OneTo(n)), src_data)
    zero!(view(dest_data,m+1:size(dest_data,1),:))
    zero!(view(dest_data,1:m,n+1:size(dest_data,2)))
    dest
end

for AbsMatOrVec in (:AbstractVector, :AbstractMatrix)
    @eval function copyto!_layout(::AbstractPaddedLayout, ::ZerosLayout, dest::$AbsMatOrVec, src::$AbsMatOrVec)
        axes(dest) == axes(src) || error("copyto! with padded/zeros only supported with equal axes")
        zero!(paddeddata(dest))
        dest
    end
end

function zero!(::AbstractPaddedLayout, A)
    zero!(paddeddata(A))
    A
end

function ArrayLayouts._norm(::AbstractPaddedLayout, A, p)
    dat = paddeddata(A)
    if MemoryLayout(dat) isa AbstractPaddedLayout
        Base.invoke(norm, Tuple{Any,Real}, dat, p)
    else
        norm(dat, p)
    end
end


# special case handle broadcasting with padded and cached arrays
function _paddedpadded_broadcasted(op, A::AbstractVector, B::AbstractVector)
    a,b = paddeddata(A),paddeddata(B)
    n,m = length(a),length(b)
    dat = if n ≤ m
        [broadcast(op, a, _view_vcat(b,1:n)); broadcast(op, zero(eltype(A)), _view_vcat(b,n+1:m))]
    else
        [broadcast(op, _view_vcat(a,1:m), b); broadcast(op, _view_vcat(a,m+1:n), zero(eltype(B)))]
    end
    CachedArray(dat, broadcast(op, Zeros{eltype(A)}(length(A)), Zeros{eltype(B)}(length(B))))
end

function _paddedpadded_broadcasted(op, A::AbstractMatrix{T}, B::AbstractMatrix{V}) where {T,V}
    a,b = paddeddata(A),paddeddata(B)
    (a_m,a_n) = size(a)
    (b_m,b_n) = size(b)
    m,n = min(a_m,b_m),min(a_n,b_n)
    dat = if a_m ≤ b_m && a_n ≤ b_n
        [broadcast(op, a, _view_vcat(b,1:a_m,1:a_n)) broadcast(op, zero(T), _view_vcat(b,1:a_m,a_n+1:b_n));
         broadcast(op, zero(T), _view_vcat(b,a_m+1:b_m,1:b_n))]
    elseif a_m ≤ b_m
        [broadcast(op, _view_vcat(a,1:a_m,1:b_n), _view_vcat(b,1:a_m,1:b_n)) broadcast(op, _view_vcat(a,1:a_m,b_n+1:a_n), zero(V));
         broadcast(op, zero(T), _view_vcat(b,a_m+1:b_m,1:b_n)) broadcast(op, Zeros{T}(b_m-a_m, a_n-b_n), Zeros{V}(b_m-a_m, a_n-b_n))]
    elseif b_n ≤ a_n # && b_m < a_m
        [broadcast(op, view(a,1:b_m,1:b_n), _view_vcat(b,1:b_m,1:b_n)) broadcast(op, _view_vcat(a,1:b_m,b_n+1:a_n), zero(V));
         broadcast(op, _view_vcat(a,b_m+1:a_m,1:a_n), zero(V))]
    else  # b_m < a_m && a_n < b_n
        [broadcast(op, _view_vcat(a,1:b_m,1:a_n), _view_vcat(b,1:b_m,1:a_n)) broadcast(op, zero(T),  _view_vcat(b,1:b_m,a_n+1:b_n));
         broadcast(op, _view_vcat(a,b_m+1:a_m,1:a_n), zero(V)) broadcast(op, Zeros{T}(a_m-b_m, b_n-a_n), Zeros{V}(a_m-b_m, b_n-a_n))]
    end
    PaddedArray(dat, max.(size(A), size(B))...)
end

layout_broadcasted(::AbstractPaddedLayout, ::AbstractPaddedLayout, op, A::AbstractVector, B::AbstractVector) =
    _paddedpadded_broadcasted(op, A, B)
layout_broadcasted(::AbstractPaddedLayout, ::AbstractPaddedLayout, op, A::AbstractMatrix, B::AbstractMatrix) =
    _paddedpadded_broadcasted(op, A, B)
layout_broadcasted(::AbstractPaddedLayout, ::AbstractPaddedLayout, ::typeof(*), A::Vcat{<:Any,1}, B::AbstractVector) =
    _paddedpadded_broadcasted(*, A, B)
layout_broadcasted(::AbstractPaddedLayout, ::AbstractPaddedLayout, ::typeof(*), A::AbstractVector, B::Vcat{<:Any,1}) =
    _paddedpadded_broadcasted(*, A, B)

function layout_broadcasted(_, ::AbstractPaddedLayout, op, A::AbstractVector, B::AbstractVector)
    b = paddeddata(B)
    m = length(b)
    zB = Zeros{eltype(B)}(size(B)...)
    CachedArray(convert(Array,broadcast(op, view(A,1:m), b)), broadcast(op, A, zB))
end

function layout_broadcasted(::AbstractPaddedLayout, _, op, A::AbstractVector, B::AbstractVector)
    a = paddeddata(A)
    n = length(a)
    zA = Zeros{eltype(A)}(size(A)...)
    CachedArray(convert(Array,broadcast(op, a, view(B,1:n))), broadcast(op, zA, B))
end

function layout_broadcasted(::AbstractPaddedLayout, ::CachedLayout, op, A::AbstractVector, B::AbstractVector)
    a = paddeddata(A)
    n = length(a)
    resizedata!(B,n)
    Bdata = cacheddata(B)
    b = view(Bdata,1:n)
    zA1 = Zeros{eltype(A)}(size(Bdata,1)-n)
    zA = Zeros{eltype(A)}(size(A)...)
    CachedArray([broadcast(op, a, b); broadcast(op, zA1, @view(Bdata[n+1:end]))], broadcast(op, zA, B.array))
end

function layout_broadcasted(::CachedLayout, ::AbstractPaddedLayout, op, A::AbstractVector, B::AbstractVector)
    b = paddeddata(B)
    n = length(b)
    resizedata!(A,n)
    Adata = cacheddata(A)
    a = view(Adata,1:n)
    zB1 = Zeros{eltype(B)}(size(Adata,1)-n)
    zB = Zeros{eltype(B)}(size(B)...)
    CachedArray([broadcast(op, a, b); broadcast(op, @view(Adata[n+1:end]), zB1)], broadcast(op, A.array, zB))
end

layout_broadcasted(lay::ApplyLayout{typeof(vcat)}, ::AbstractPaddedLayout, op, A::AbstractVector, B::AbstractVector) =
    layout_broadcasted(lay, lay, op, A, B)
layout_broadcasted(::AbstractPaddedLayout, lay::ApplyLayout{typeof(vcat)}, op, A::AbstractVector, B::AbstractVector) =
    layout_broadcasted(lay, lay, op, A, B)


# special case for * to preserve Vcat Structure

layout_broadcasted(::ApplyLayout{typeof(vcat)}, lay::AbstractPaddedLayout, ::typeof(*), A::AbstractVector, B::AbstractVector) =
    layout_broadcasted(UnknownLayout(), lay, *, A, B)
layout_broadcasted(lay::AbstractPaddedLayout, ::ApplyLayout{typeof(vcat)}, ::typeof(*), A::AbstractVector, B::AbstractVector) =
    layout_broadcasted(lay, UnknownLayout(), *, A, B)

for op in (:+, :-)
    @eval function layout_broadcasted(::AbstractPaddedLayout, ::AbstractPaddedLayout, ::typeof($op), A::Vcat{T,1}, B::Vcat{V,1}) where {T,V}
        a,b = paddeddata(A),paddeddata(B)
        if a isa Number && b isa Number
            Vcat($op(a, b), Zeros{promote_type(T,V)}(max(length(A),length(B))-1))
        elseif a isa Number
            m = length(b)
            dat = [broadcast($op, a, b[1]); broadcast($op,view(b,2:m))]
            Vcat(convert(Array,dat), Zeros{promote_type(T,V)}(max(length(A),length(B))-length(dat)))
        elseif b isa Number
            n = length(a)
            dat = [broadcast($op, a[1], b); view(a,2:n)]
            Vcat(convert(Array,dat), Zeros{promote_type(T,V)}(max(length(A),length(B))-length(dat)))
        else
            n,m = length(a),length(b)
            dat = if n == m
                broadcast($op, a, b)
            elseif n > m
                [broadcast($op, view(a,1:m), b); view(a,m+1:n)]
            else # n < m
                [broadcast($op, a, view(b,1:n)); broadcast($op,view(b,n+1:m))]
            end
            Vcat(convert(Array,dat), Zeros{promote_type(T,V)}(max(length(A),length(B))-length(dat)))
        end
    end

    @eval function layout_broadcasted(::AbstractPaddedLayout, ::AbstractPaddedLayout, ::typeof($op), A::Vcat{T,2}, B::Vcat{V,2}) where {T,V}
        a,b = paddeddata(A),paddeddata(B)
        n,m = size(a,1),size(b,1)
        dat = if n == m
            broadcast($op, a, b)
        elseif n > m
            [broadcast($op, view(a,1:m,:), b); view(a,m+1:n,:)]
        else # n < m
            [broadcast($op, a, view(b,1:n,:)); broadcast($op,view(b,n+1:m,:))]
        end
        Vcat(convert(Array,dat), Zeros{promote_type(T,V)}(max(size(A,1),size(B,1))-size(dat,1),size(a,2)))
    end
end

function layout_broadcasted(::AbstractPaddedLayout, ::AbstractPaddedLayout, ::typeof(*), A::Vcat{T,1}, B::Vcat{V,1}) where {T,V}
    a,b = paddeddata(A),paddeddata(B)
    if a isa Number || b isa Number
        Vcat(a[1]*b[1], Zeros{promote_type(T,V)}(max(length(A),length(B))-1))
    else
        n = min(length(a),length(b))
        dat = broadcast(*, view(a,1:n), view(b,1:n))
        Vcat(convert(Array,dat), Zeros{promote_type(T,V)}(max(length(A),length(B))-n))
    end
end

function layout_broadcasted(_, ::AbstractPaddedLayout, ::typeof(*), A::AbstractVector{T}, B::Vcat{V,1}) where {T,V}
    b = paddeddata(B)
    m = length(b)
    dat = broadcast(*, view(A,1:m), b)
    Vcat(convert(Array,dat), Zeros{promote_type(T,V)}(max(length(A),length(B))-m))
end


function layout_broadcasted(::AbstractPaddedLayout, _, ::typeof(*), A::Vcat{T,1}, B::AbstractVector{V}) where {T,V}
    a = paddeddata(A)
    n = length(a)
    dat = broadcast(*, a, view(B,1:n))
    Vcat(convert(Array,dat), Zeros{promote_type(T,V)}(max(length(A),length(B))-n))
end

layout_broadcasted(::AbstractPaddedLayout, lay::ApplyLayout{typeof(vcat)}, ::typeof(*), A::Vcat{<:Any,1}, B::AbstractVector) = layout_broadcasted(lay, lay, *, A, B)
layout_broadcasted(lay::ApplyLayout{typeof(vcat)}, ::AbstractPaddedLayout, ::typeof(*), A::AbstractVector, B::Vcat{<:Any,1}) = layout_broadcasted(lay, lay, *, A, B)


layout_broadcasted(_, _, op, A, B) = Base.Broadcast.Broadcasted{typeof(Base.BroadcastStyle(Base.BroadcastStyle(typeof(A)),Base.BroadcastStyle(typeof(B))))}(op, (A, B))

###
# Dot/Axpy
###


struct Axpy{StyleX,StyleY,T,XTyp,YTyp}
    α::T
    X::XTyp
    Y::YTyp
end

Axpy(α::T, X::XTyp, Y::YTyp) where {T,XTyp,YTyp} = Axpy{typeof(MemoryLayout(XTyp)), typeof(MemoryLayout(YTyp)), T, XTyp, YTyp}(α, X, Y)
materialize!(d::Axpy{<:Any,<:Any,<:Number,<:AbstractArray,<:AbstractArray}) = Base.invoke(LinearAlgebra.axpy!, Tuple{Number,AbstractArray,AbstractArray}, d.α, d.X, d.Y)
function materialize!(d::Axpy{<:PaddedColumns,<:PaddedColumns,U,<:AbstractVector{T},<:AbstractVector{V}}) where {U,T,V}
    x = paddeddata(d.X)
    resizedata!(d.Y, length(x))
    y = paddeddata(d.Y)
    axpy!(d.α, x, view(y,1:length(x)))
    y
end
axpy!(α, X, Y) = materialize!(Axpy(α,X,Y))
LinearAlgebra.axpy!(α, X::LazyArray, Y::AbstractArray) = materialize!(Axpy(α, X, Y))
LinearAlgebra.axpy!(α, X::SubArray{<:Any,N,<:LazyArray}, Y::AbstractArray) where {N} = materialize!(Axpy(α, X, Y))


###
# l/rmul!
###

function materialize!(M::Lmul{ScalarLayout,<:AbstractPaddedLayout})
    lmul!(M.A, paddeddata(M.B))
    M.B
end

function materialize!(M::Rmul{<:AbstractPaddedLayout,ScalarLayout})
    rmul!(paddeddata(M.A), M.B)
    M.A
end


_norm2(::AbstractPaddedLayout, a) = norm(paddeddata(a),2)
_norm1(::AbstractPaddedLayout, a) = norm(paddeddata(a),1)
_normInf(::AbstractPaddedLayout, a) = norm(paddeddata(a),Inf)
_normp(::AbstractPaddedLayout, a, p) = norm(paddeddata(a),p)


for (Dt, dt) in ((:Dot, :dot), (:Dotu, :dotu))
    @eval begin
        function copy(D::$Dt{layA, layB}) where {layA<:AbstractPaddedLayout,layB<:AbstractPaddedLayout}
            a = paddeddata(D.A)
            b = paddeddata(D.B)
            T = eltype(D)
            if MemoryLayout(a) isa layA && MemoryLayout(b) isa layB
                return convert(T, $dt(Array(a),Array(b)))
            end
            length(a) == length(b) && return convert(T, $dt(a,b))
            # following handles scalars
            ((length(a) == 1) || (length(b) == 1)) && return convert(T, a[1] * b[1])
            m = min(length(a), length(b))
            convert(T, $dt(view(a, 1:m), view(b, 1:m)))
        end

        function copy(D::$Dt{<:AbstractPaddedLayout})
            a = paddeddata(D.A)
            m = length(a)
            v = view(D.B, 1:m)
            if MemoryLayout(a) isa AbstractPaddedLayout
                convert(eltype(D), $dt(Array(a), v))
            else
                convert(eltype(D), $dt(a, v))
            end

        end

        function copy(D::$Dt{<:Any, <:AbstractPaddedLayout})
            b = paddeddata(D.B)
            m = length(b)
            v = view(D.A, 1:m)
            if MemoryLayout(b) isa AbstractPaddedLayout
                convert(eltype(D), $dt(v, Array(b)))
            else
                convert(eltype(D), $dt(v, b))
            end
        end
    end
end

_vcat_sub_arguments(::AbstractPaddedLayout, A, V) = _vcat_sub_arguments(ApplyLayout{typeof(vcat)}(), A, V)

_lazy_getindex(dat, kr...) = view(dat, kr...)
_lazy_getindex(dat::Number, kr) = 1 ∈ kr ? dat : zero(dat)
_lazy_getindex(dat::Number, kr, jr) = 1 ∈ kr && 1 ∈ jr ? dat : zero(dat)

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
    if MemoryLayout(dat) isa AbstractPaddedLayout
        Vcat(dat, Zeros{T}(length(v) - length(dat)))
    else
        Vcat(sub_materialize(dat), Zeros{T}(length(v) - length(dat)))
    end
end

sub_materialize(::AbstractPaddedLayout, v::AbstractVector, _) = _padded_sub_materialize(v)

function sub_materialize(l::AbstractPaddedLayout, v::AbstractMatrix, _)
    dat = paddeddata(v)
    PaddedArray(MemoryLayout(dat) isa PaddedLayout ? dat : sub_materialize(dat), size(v)...)
end

function layout_replace_in_print_matrix(::AbstractPaddedLayout{Lay}, f::AbstractVecOrMat, k, j, s) where Lay
    # avoid infinite-loop
    f isa SubArray && return Base.replace_in_print_matrix(parent(f), Base.reindex(f.indices, (k,j))..., s)
    data = paddeddata(f)
    MemoryLayout(data) isa AbstractPaddedLayout{Lay} && return layout_replace_in_print_matrix(UnknownLayout(), f, k, j, s)
    k in axes(data,1) && j in axes(data,2) && return _replace_in_print_matrix(data, k, j, s)
    Base.replace_with_centered_mark(s)
end

# avoid ambiguity in LazyBandedMatrices
copy(M::Mul{<:DiagonalLayout,<:Union{PaddedColumns,PaddedLayout}}) = copy(Lmul(M))
copy(M::Mul{<:Union{TriangularLayout{'U', 'N', <:AbstractLazyLayout}, TriangularLayout{'U', 'U', <:AbstractLazyLayout}}, <:Union{PaddedColumns,PaddedLayout}}) = copy(Lmul(M))
simplifiable(::Mul{<:Union{TriangularLayout{'U', 'N', <:AbstractLazyLayout}, TriangularLayout{'U', 'U', <:AbstractLazyLayout}}, <:Union{PaddedColumns,PaddedLayout}}) = Val(true)


@inline simplifiable(M::Mul{BroadcastLayout{typeof(*)},<:Union{PaddedColumns,PaddedLayout}}) = simplifiable(Mul{BroadcastLayout{typeof(*)},UnknownLayout}(M.A,M.B))
@inline copy(M::Mul{BroadcastLayout{typeof(*)},<:Union{PaddedColumns,PaddedLayout}}) = copy(Mul{BroadcastLayout{typeof(*)},UnknownLayout}(M.A,M.B))

simplifiable(::Mul{<:DualLayout{<:AbstractLazyLayout}, <:Union{PaddedColumns,PaddedLayout}}) = Val(true)
copy(M::Mul{<:DualLayout{<:AbstractLazyLayout}, <:Union{PaddedColumns,PaddedLayout}}) = copy(mulreduce(M))
simplifiable(::Mul{<:DiagonalLayout{<:AbstractFillLayout}, <:Union{PaddedColumns,PaddedLayout}}) = Val(true)

function simplifiable(M::Mul{<:DualLayout{<:PaddedRows}, <:LazyLayouts})
    trans = transtype(M.A)
    simplifiable(*, trans(M.B), trans(M.A))
end
function copy(M::Mul{<:DualLayout{<:PaddedRows}, <:LazyLayouts})
    trans = transtype(M.A)
    trans(trans(M.B) * trans(M.A))
end

for op in (:+, :-)
    @eval begin
        simplifiable(M::Mul{BroadcastLayout{typeof($op)},<:Union{PaddedColumns,PaddedLayout}}) = Val(true)
        simplifiable(M::Mul{<:DualLayout{<:PaddedRows},BroadcastLayout{typeof($op)}}) = Val(true)
        copy(M::Mul{BroadcastLayout{typeof($op)},<:Union{PaddedColumns,PaddedLayout}}) =  copy(Mul{BroadcastLayout{typeof($op)},UnknownLayout}(M.A, M.B))
        copy(M::Mul{<:DualLayout{<:PaddedRows},BroadcastLayout{typeof($op)}}) =  copy(Mul{UnknownLayout,BroadcastLayout{typeof($op)}}(M.A, M.B))
    end
end


# Triangular columns

sublayout(::TriangularLayout{'U','N', ML}, INDS::Type{<:Tuple{KR,Integer}}) where {KR,ML} =
    sublayout(PaddedColumns{typeof(sublayout(ML(), INDS))}(), Tuple{KR})

sublayout(::TriangularLayout{'L','N', ML}, INDS::Type{<:Tuple{Integer,JR}}) where {JR,ML} =
    sublayout(PaddedColumns{typeof(sublayout(ML(), INDS))}(), Tuple{JR})


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

@inline applied_ndims(::typeof(setindex), a, b...) = ndims(a)
@inline applied_eltype(::typeof(setindex), a, b...) = eltype(a)
@inline applied_axes(::typeof(setindex), a, b...) = axes(a)
@inline applied_size(::typeof(setindex), a, b...) = size(a)

function getindex(A::ApplyVector{T,typeof(setindex)}, k::Integer) where T
    P,v,kr = A.args
    convert(T, k in kr ? v[something(findlast(isequal(k),kr))] : P[k])::T
end

function getindex(A::ApplyMatrix{T,typeof(setindex)}, k::Integer, j::Integer) where T
    P,v,kr,jr = A.args
    convert(T, k in kr && j in jr ? v[something(findlast(isequal(k),kr)),something(findlast(isequal(j),jr))] : P[k,j])::T
end

const PaddedArray{T,N,M} = ApplyArray{T,N,typeof(setindex),<:Tuple{Zeros,M,Vararg{Any,N}}}
const PaddedVector{T,M} = PaddedArray{T,1,M}
const PaddedMatrix{T,M} = PaddedArray{T,2,M}

MemoryLayout(::Type{<:PaddedVector{T,M}}) where {T,M} = PaddedColumns{typeof(MemoryLayout(M))}()
MemoryLayout(::Type{<:PaddedMatrix{T,M}}) where {T,M} = PaddedLayout{typeof(MemoryLayout(M))}()
paddeddata(A::PaddedArray) = paddeddata_axes(axes(A), A)
paddeddata_axes(_, A) = A.args[2]

PaddedArray(A::AbstractArray{T,N}, n::Vararg{Integer,N}) where {T,N} = PaddedArray(A, map(oneto,n))
PaddedArray(A::AbstractArray{T,N}, ax::NTuple{N,Any}) where {T,N} = ApplyArray{T,N}(setindex, Zeros{T,N}(ax), A, axes(A)...)
PaddedArray(A::T, n::Vararg{Integer,N}) where {T<:Number,N} = ApplyArray{T,N}(setindex, Zeros{T,N}(n...), A, ntuple(_ -> OneTo(1),N)...)
(PaddedArray{T,N} where T)(A, n::Vararg{Integer,N}) where N = PaddedArray(A, n...)
PaddedVector(A::AbstractVector{T}, ax::AbstractUnitRange) where T = ApplyArray{T, 1}(setindex, Zeros{T, 1}((ax, )), A, axes(A)...)
PaddedMatrix(A::AbstractMatrix{T}, ax::NTuple{2, Any}) where T = PaddedArray(A, ax)

BroadcastStyle(::Type{<:PaddedArray{<:Any,N}}) where N = LazyArrayStyle{N}()



function ArrayLayouts._bidiag_forwardsub!(M::Ldiv{<:Any,<:PaddedColumns,<:AbstractMatrix,<:AbstractVector})
    A, b_in = M.A, M.B
    dv = diagonaldata(A)
    ev = subdiagonaldata(A)
    b = paddeddata(b_in)
    N = length(b)
    dvj = dv[1]
    iszero(dvj) && throw(SingularException(1))
    b[1] = bj1 = dvj\b[1]
    @inbounds for j = 2:N
        bj  = b[j]
        bj -= ev[j - 1] * bj1
        dvj = dv[j]
        iszero(dvj) && throw(SingularException(j))
        bj   = dvj\bj
        b[j] = bj1 = bj
    end

    @inbounds for j = N+1:length(b_in)
        iszero(bj1) && break
        bj = -ev[j - 1] * bj1
        dvj = dv[j]
        iszero(dvj) && throw(SingularException(j))
        bj   = dvj\bj
        b_in[j] = bj1 = bj
    end

    b_in
end
