struct LazyArrayStyle{N} <: AbstractArrayStyle{N} end
LazyArrayStyle(::Val{N}) where N = LazyArrayStyle{N}()
LazyArrayStyle{M}(::Val{N}) where {N,M} = LazyArrayStyle{N}()
"""
    BroadcastLayout{F}()

is returned by `MemoryLayout(A)` if a matrix `A` is a `BroadcastArray`.
`F` is the typeof function that broadcast operation is applied.
"""
struct BroadcastLayout{F} <: AbstractLazyLayout end

@inline tuple_type_memorylayouts(::Type{Tuple{}}) = ()
@inline tuple_type_memorylayouts(::Type{I}) where I<:Tuple = tuple(MemoryLayout(Base.tuple_type_head(I)), tuple_type_memorylayouts(Base.tuple_type_tail(I))...)
@inline tuple_type_memorylayouts(::Type{Tuple{A}}) where {A} = (MemoryLayout(A),)
@inline tuple_type_memorylayouts(::Type{Tuple{A,B}}) where {A,B} = (MemoryLayout(A),MemoryLayout(B))
@inline tuple_type_memorylayouts(::Type{Tuple{A,B,C}}) where {A,B,C} = (MemoryLayout(A),MemoryLayout(B),MemoryLayout(C))
@inline tuple_type_memorylayouts(::Type{Tuple{A,B,C,D}}) where {A,B,C,D} = (MemoryLayout(A),MemoryLayout(B),MemoryLayout(C),MemoryLayout(D))
@inline tuple_type_memorylayouts(::Type{Tuple{A,B,C,D,E}}) where {A,B,C,D,E} = (MemoryLayout(A),MemoryLayout(B),MemoryLayout(C),MemoryLayout(D),MemoryLayout(E))

@inline broadcastlayout(::Type{F}, _...) where F = BroadcastLayout{F}()


function copyto!_layout(_, ::BroadcastLayout, dest::AbstractArray{<:Any,N}, bc::AbstractArray{<:Any,N}) where N
    materialize!(dest, _broadcastarray2broadcasted(bc))
    dest
end

struct BroadcastArray{T, N, F, Args} <: LazyArray{T, N}
    f::F
    args::Args
end

const BroadcastVector{T,F,Args} = BroadcastArray{T,1,F,Args}
const BroadcastMatrix{T,F,Args} = BroadcastArray{T,2,F,Args}

LazyArray(bc::Broadcasted) = BroadcastArray(bc)

BroadcastArray{T,N,F,Args}(bc::Broadcasted) where {T,N,F,Args} = BroadcastArray{T,N,F,Args}(bc.f,bc.args)
BroadcastArray{T,N}(bc::Broadcasted{Style,Axes,F,Args}) where {T,N,Style,Axes,F,Args} = BroadcastArray{T,N,F,Args}(bc.f,bc.args)
BroadcastArray{T}(bc::Broadcasted{<:Union{Nothing,BroadcastStyle},<:Tuple{Vararg{Any,N}},<:Any,<:Tuple}) where {T,N} =
    BroadcastArray{T,N}(bc)

BroadcastVector(bc::Broadcasted) = BroadcastVector{combine_eltypes(bc.f, bc.args)}(bc)
BroadcastMatrix(bc::Broadcasted) = BroadcastMatrix{combine_eltypes(bc.f, bc.args)}(bc)

@inline MemoryLayout(::Type{BroadcastArray{T,N,F,Args}}) where {T,N,F,Args} =
    broadcastlayout(F, tuple_type_memorylayouts(Args)...)

arguments(::BroadcastLayout{F}, A::BroadcastArray{<:Any,N,F}) where {N,F} = A.args

_broadcast2broadcastarray() = ()
_broadcast2broadcastarray(a, b...) = tuple(a, _broadcast2broadcastarray(b...)...)
_broadcast2broadcastarray(a::Broadcasted{DefaultArrayStyle{0}}, b...) = tuple(materialize(a), _broadcast2broadcastarray(b...)...)
_broadcast2broadcastarray(a::Broadcasted, b...) = tuple(BroadcastArray(a), _broadcast2broadcastarray(b...)...)

_BroadcastArray(bc::Broadcasted) = BroadcastArray{combine_eltypes(bc.f, bc.args)}(bc)
BroadcastArray(bc::Broadcasted{S}) where S =
    _BroadcastArray(instantiate(Broadcasted{S}(bc.f, _broadcast2broadcastarray(bc.args...))))

BroadcastArray(f, A, As...) = BroadcastArray(broadcasted(f, A, As...))
BroadcastArray{T}(f, A, As...) where T = BroadcastArray{T}(instantiate(broadcasted(f, A, As...)))
BroadcastMatrix(f, A...) = BroadcastMatrix(broadcasted(f, A...))
BroadcastVector(f, A...) = BroadcastVector(broadcasted(f, A...))

BroadcastArray{T,N}(f, A...) where {T,N} = BroadcastArray{T,N,typeof(f),typeof(A)}(f, A)

BroadcastArray(b::BroadcastArray) = b
BroadcastVector(A::BroadcastVector) = A
BroadcastMatrix(A::BroadcastMatrix) = A

@inline __broadcastarray2broadcasted() = ()
@inline __broadcastarray2broadcasted(a, b...) = tuple(_broadcastarray2broadcasted(a), __broadcastarray2broadcasted(b...)...)
@inline _broadcastarray2broadcasted(lay::BroadcastLayout, a) = broadcasted(call(lay, a), __broadcastarray2broadcasted(arguments(lay, a)...)...)
@inline _broadcastarray2broadcasted(lay::BroadcastLayout, a::BroadcastArray) = broadcasted(call(lay, a), __broadcastarray2broadcasted(arguments(lay, a)...)...)
@inline _broadcastarray2broadcasted(_, a) = a
@inline _broadcastarray2broadcasted(lay, a::BroadcastArray) = error("Overload LazyArrays._broadcastarray2broadcasted(::$(lay), _)")
@inline _broadcastarray2broadcasted(::DualLayout{ML}, a) where ML = _broadcastarray2broadcasted(ML(), a)
@inline _broadcastarray2broadcasted(::DualLayout{ML}, a::BroadcastArray) where ML = _broadcastarray2broadcasted(ML(), a)
@inline _broadcastarray2broadcasted(a) = _broadcastarray2broadcasted(MemoryLayout(a), a)
@inline _broadcasted(A) = instantiate(_broadcastarray2broadcasted(A))
broadcasted(A::BroadcastArray) = _broadcasted(A)
broadcasted(A::SubArray{<:Any,N,<:BroadcastArray}) where N = _broadcasted(A)
Broadcasted(A::BroadcastArray) = broadcasted(A)::Broadcasted
Broadcasted(A::SubArray{<:Any,N,<:BroadcastArray}) where N = broadcasted(A)::Broadcasted

@inline BroadcastArray(A::AbstractArray) = BroadcastArray(call(A), arguments(A)...)

axes(A::BroadcastArray) = axes(broadcasted(A))
size(A::BroadcastArray) = map(length, axes(A))

_broadcast_last(a) = a
_broadcast_last(a::AbstractArray) = last(a)
_broadcast_last(a::Ref) = a[]
last(A::BroadcastArray) = A.f(_broadcast_last.(A.args)...)

@propagate_inbounds getindex(A::BroadcastArray{T,N}, kj::Vararg{Int,N}) where {T,N} = convert(T,broadcasted(A)[kj...])::T

converteltype(::Type{T}, A::AbstractArray) where T = convert(AbstractArray{T}, A)
converteltype(::Type{T}, A) where T = convert(T, A)
sub_materialize(::BroadcastLayout, A) = converteltype(eltype(A), sub_materialize(_broadcasted(A)))

copy(bc::Broadcasted{<:LazyArrayStyle}) = BroadcastArray(bc)

# BroadcastArray are immutable
copy(bc::BroadcastArray) = bc
map(::typeof(copy), bc::BroadcastArray) = bc
copy(bc::AdjOrTrans{<:Any,<:BroadcastArray}) = bc

# Replacement for #18.
# Could extend this to other similar reductions in Base... or apply at lower level?
# for (fname, op) in [(:sum, :add_sum), (:prod, :mul_prod),
#                     (:maximum, :max), (:minimum, :min),
#                     (:all, :&),       (:any, :|)]
function Base._sum(f, A::BroadcastArray, ::Colon)
    bc = broadcasted(A)
    T = Broadcast.combine_eltypes(f ∘ bc.f, bc.args)
    out = zero(T)
    @simd for I in eachindex(bc)
        @inbounds out += f(bc[I])
    end
    out
end
function Base._prod(f, A::BroadcastArray, ::Colon)
    bc = broadcasted(A)
    T = Broadcast.combine_eltypes(f ∘ bc.f, bc.args)
    out = one(T)
    @simd for I in eachindex(bc)
        @inbounds out *= f(bc[I])
    end
    out
end


BroadcastStyle(::Type{<:LazyArray{<:Any,N}}) where N = LazyArrayStyle{N}()
BroadcastStyle(::Type{<:Adjoint{<:Any,<:LazyVector}}) = LazyArrayStyle{2}()
BroadcastStyle(::Type{<:Transpose{<:Any,<:LazyVector}}) = LazyArrayStyle{2}()
BroadcastStyle(::Type{<:Adjoint{<:Any,<:LazyMatrix}}) = LazyArrayStyle{2}()
BroadcastStyle(::Type{<:Transpose{<:Any,<:LazyMatrix}}) = LazyArrayStyle{2}()
BroadcastStyle(::Type{<:SubArray{<:Any,1,<:LazyMatrix,<:Tuple{Slice,Any}}}) = LazyArrayStyle{1}()

BroadcastStyle(::Type{<:UpperOrLowerTriangular{<:Any,<:LazyMatrix}}) = LazyArrayStyle{2}()
BroadcastStyle(::Type{<:LinearAlgebra.HermOrSym{<:Any,<:LazyMatrix}}) = LazyArrayStyle{2}()


BroadcastStyle(L::LazyArrayStyle{N}, ::StructuredMatrixStyle)  where N = L



## scalar-range broadcast operations ##
# Ranges already support smart broadcasting
for op in (+, -, big)
    @eval begin
        broadcasted(::LazyArrayStyle{1}, ::typeof($op), r::AbstractRange) =
            broadcast(DefaultArrayStyle{1}(), $op, r)
    end
end

for op in (-, +, *, /)
    @eval broadcasted(::LazyArrayStyle{1}, ::typeof($op), r::AbstractRange, x::Real) = broadcast(DefaultArrayStyle{1}(), $op, r, x)
end

for op in (-, +, *, \)
    @eval broadcasted(::LazyArrayStyle{1}, ::typeof($op), x::Real, r::AbstractRange) = broadcast(DefaultArrayStyle{1}(), $op, x, r)
end

broadcasted(::LazyArrayStyle{N}, op, r::AbstractFill{T,N}) where {T,N} = broadcast(DefaultArrayStyle{N}(), op, r)
broadcasted(::LazyArrayStyle{N}, op, r::AbstractFill{T,N}, x::Number) where {T,N} = broadcast(DefaultArrayStyle{N}(), op, r, x)
broadcasted(::LazyArrayStyle{N}, op, x::Number, r::AbstractFill{T,N}) where {T,N} = broadcast(DefaultArrayStyle{N}(), op, x, r)
broadcasted(::LazyArrayStyle{N}, op, r::AbstractFill{T,N}, x::Ref) where {T,N} = broadcast(DefaultArrayStyle{N}(), op, r, x)
broadcasted(::LazyArrayStyle{N}, op, x::Ref, r::AbstractFill{T,N}) where {T,N} = broadcast(DefaultArrayStyle{N}(), op, x, r)
broadcasted(::LazyArrayStyle{N}, op, r1::AbstractFill{T,N}, r2::AbstractFill{V,N}) where {T,V,N} = broadcast(DefaultArrayStyle{N}(), op, r1, r2)
broadcasted(::LazyArrayStyle{1}, ::typeof(*), a::AbstractFill, b::AbstractRange) = broadcast(DefaultArrayStyle{1}(), *, a, b)
broadcasted(::LazyArrayStyle{1}, ::typeof(*), a::AbstractRange, b::AbstractFill) = broadcast(DefaultArrayStyle{1}(), *, a, b)
broadcasted(::LazyArrayStyle{1}, ::typeof(*), a::Zeros{<:Any,1}, b::AbstractRange) = broadcast(DefaultArrayStyle{1}(), *, a, b)
broadcasted(::LazyArrayStyle{1}, ::typeof(*), a::AbstractRange, b::Zeros{<:Any,1}) = broadcast(DefaultArrayStyle{1}(), *, a, b)

for op in (:*, :/, :\)
    @eval broadcasted(::LazyArrayStyle{N}, ::typeof($op), a::Zeros{T,N}, b::Zeros{V,N}) where {T,V,N} = broadcast(DefaultArrayStyle{N}(), $op, a, b)
end

for op in (:*, :/)
    @eval begin
        broadcasted(::LazyArrayStyle{N}, ::typeof($op), a::Zeros{T,N}, b::AbstractArray{V,N}) where {T,V,N} = broadcast(DefaultArrayStyle{N}(), $op, a, b)
        broadcasted(::LazyArrayStyle{N}, ::typeof($op), a::Zeros{T,N}, b::Broadcasted) where {T,N} = broadcast(DefaultArrayStyle{N}(), $op, a, b)
    end
end
for op in (:*, :\)
    @eval begin
        broadcasted(::LazyArrayStyle{N}, ::typeof($op), a::AbstractArray{T,N}, b::Zeros{V,N}) where {T,V,N} = broadcast(DefaultArrayStyle{N}(), $op, a, b)
        broadcasted(::LazyArrayStyle{N}, ::typeof($op), a::Broadcasted, b::Zeros{V,N}) where {V,N} = broadcast(DefaultArrayStyle{N}(), $op, a, b)
    end
end

###
# support
###

_broadcast_colsupport(ax, ::Tuple{}, A, j) = ax[1]
_broadcast_colsupport(ax, ::Tuple{<:Any}, A, j) = colsupport(A,j)
_broadcast_colsupport(ax, Aax::Tuple{OneTo{Int},<:Any}, A, j) = length(Aax[1]) == 1 ? ax[1] : colsupport(A,j)
_broadcast_colsupport(ax, ::Tuple{<:Any,<:Any}, A, j) = colsupport(A,j)
_broadcast_rowsupport(ax, ::Tuple{}, A, j) = ax[2]
_broadcast_rowsupport(ax, ::Tuple{<:Any}, A, j) = ax[2]
_broadcast_rowsupport(ax, Aax::Tuple{<:Any,OneTo{Int}}, A, j) = length(Aax[2]) == 1 ? ax[2] : rowsupport(A,j)
_broadcast_rowsupport(ax, ::Tuple{<:Any,<:Any}, A, j) = rowsupport(A,j)

colsupport(lay::BroadcastLayout{typeof(*)}, A, j) = intersect(_broadcast_colsupport.(Ref(axes(A)), axes.(arguments(lay,A)), arguments(lay,A), Ref(j))...)
rowsupport(lay::BroadcastLayout{typeof(*)}, A, j) = intersect(_broadcast_rowsupport.(Ref(axes(A)), axes.(arguments(lay,A)), arguments(lay,A), Ref(j))...)

function colsupport(lay::BroadcastLayout{typeof(\)}, A, j)
    _,b = arguments(lay,A)
    _broadcast_colsupport(axes(A), axes(b), b, j)
end

function rowsupport(lay::BroadcastLayout{typeof(\)}, A, j)
    _,b = arguments(lay,A)
    _broadcast_rowsupport(axes(A), axes(b), b, j)
end

function colsupport(lay::BroadcastLayout{typeof(/)}, A, j)
    b,_ = arguments(lay,A)
    _broadcast_colsupport(axes(A), axes(b), b, j)
end

function rowsupport(lay::BroadcastLayout{typeof(/)}, A, j)
    b,_ = arguments(lay,A)
    _broadcast_rowsupport(axes(A), axes(b), b, j)
end

for op in (:+, :-)
    @eval begin
        rowsupport(lay::BroadcastLayout{typeof($op)}, A, j) = convexunion(_broadcast_rowsupport.(Ref(axes(A)), axes.(arguments(lay,A)), arguments(lay,A), Ref(j))...)
        colsupport(lay::BroadcastLayout{typeof($op)}, A, j) = convexunion(_broadcast_colsupport.(Ref(axes(A)), axes.(arguments(lay,A)), arguments(lay,A), Ref(j))...)
    end
end


###
# SubArray
###

# TODO: special case adjtrans to skip the `isone` check and return numbers instead of 1-vectors.
#

sublayout(b::BroadcastLayout, _) = b

@inline _broadcastviewinds(::Tuple{}, inds) = ()
@inline _broadcastviewinds(ax::Tuple{OneTo{Int},Vararg{Any}}, inds::Tuple{Number,Vararg{Any}}) =
    tuple(isone(length(ax[1])) ? 1 : convert(Int,inds[1]), _broadcastviewinds(tail(ax), tail(inds))...)
@inline _broadcastviewinds(ax::Tuple{OneTo{Int},Vararg{Any}}, inds::Tuple{AbstractVector{<:Integer},Vararg{Any}}) =
    tuple(isone(length(ax[1])) ? convert(typeof(inds[1]),Base.OneTo(min(1,length(inds[1])))) : inds[1], _broadcastviewinds(tail(ax), tail(inds))...)
@inline function _broadcastviewinds(ax::Tuple{OneTo{Int},Vararg{Any}}, inds::Tuple{Any,Vararg{Any}})
    @assert isone(length(ax[1]))
    tuple(Base.OneTo(1), _broadcastviewinds(tail(ax), tail(inds))...)
end

@inline _broadcastviewinds(ax, inds) = # don't support special broadcasting
    tuple(inds[1], _broadcastviewinds(tail(ax), tail(inds))...)

_viewifmutable(a, inds::Number...) = a[inds...]
@inline _viewifmutable(a, inds...) = view(a, inds...)
@inline _viewifmutable(a::AbstractFill, inds...) = a[inds...]
@inline _viewifmutable(a::AbstractRange, inds...) = a[inds...]
@inline _viewifmutable(a::AbstractRange, inds::Number...) = a[inds...]
# _viewifmutable(a::BroadcastArray, inds...) = a[inds...]
_viewifmutable(a::AdjOrTrans{<:Any,<:AbstractVector}, k::Integer, j::Integer) = a[k,j]
function _viewifmutable(a::AdjOrTrans{<:Any,<:AbstractVector}, k::Integer, j)
    @assert k == 1
    _viewifmutable(parent(a), j)
end
@inline _broadcastview(a, inds) = _viewifmutable(a, _broadcastviewinds(axes(a), inds)...)
@inline _broadcastview(a::Number, inds) = a
@inline _broadcastview(a::Base.RefValue, inds) = a

@inline __broadcastview(inds) = ()
@inline __broadcastview(inds, a, b...) = (_broadcastview(a, inds), __broadcastview(inds, b...)...)

@inline function _broadcast_sub_arguments(lay, P, V)
    args = arguments(lay, P)
    __broadcastview(parentindices(V), args...)
end

@inline _broadcast_sub_arguments(lay::DualLayout{ML}, P, V::AbstractVector) where ML =
    arguments(ML(), view(_adjortrans(P), parentindices(V)[2]))

@inline _broadcast_sub_arguments(A, V) = _broadcast_sub_arguments(MemoryLayout(A), A, V)
@inline _broadcast_sub_arguments(V) =  _broadcast_sub_arguments(parent(V), V)
@inline arguments(lay::BroadcastLayout, V::SubArray) = _broadcast_sub_arguments(V)
@inline call(b::BroadcastLayout, a::SubArray) = call(b, parent(a))


###
# Transpose
###

call(b::BroadcastLayout, a::AdjOrTrans) = call(b, parent(a))

transposelayout(b::BroadcastLayout) = b

_adjoint(a) = adjoint(a)
_adjoint(a::Ref) = a
_transpose(a) = transpose(a)
_transpose(a::Ref) = a

arguments(b::BroadcastLayout, A::Adjoint) = map(_adjoint, arguments(b, parent(A)))
arguments(b::BroadcastLayout, A::Transpose) = map(_transpose, arguments(b, parent(A)))

# broadcasting a transpose is the same as broadcasting it to the array and transposing
# this allows us to collapse to one broadcast.
broadcasted(::LazyArrayStyle, op, A::Transpose{<:Any,<:BroadcastArray}) = transpose(broadcast(op, parent(A)))
broadcasted(::LazyArrayStyle, op, A::Adjoint{<:Real,<:BroadcastArray}) = adjoint(broadcast(op, parent(A)))

# ensure we benefit from fast linear indexing
getindex(A::Transpose{<:Any,<:BroadcastVector}, k::AbstractVector) = parent(A)[k]
getindex(A::Adjoint{<:Real,<:BroadcastVector}, k::AbstractVector) = parent(A)[k]
getindex(A::Adjoint{<:Any,<:BroadcastVector}, k::AbstractVector) = conj.(parent(A))[k]


###
# Show
###

_broadcastarray_summary(io, A) = _broadcastarray_summary(io, A.f, arguments(A)...)
function _broadcastarray_summary(io, f, args...)
    print(io, "$f.(")
    summary(io, first(args))
    for a in tail(args)
        print(io, ", ")
        summary(io, a)
    end
    print(io, ")")
end

for op in (:+, :-, :*, :\, :/)
    @eval begin
        function _broadcastarray_summary(io::IO, ::typeof($op), args...)
            if length(args) == 1
                print(io, "($($op)).(")
                summary(io, first(args))
                print(io, ")")
            else
                print(io, "(")
                summary(io, first(args))
                print(io, ")")
                for a in tail(args)
                    print(io, " .$($op) (")
                    summary(io, a)
                    print(io, ")")
                end
            end
        end
    end
end

function _broadcastarray_summary(io::IO, ::typeof(Base.literal_pow), ::Base.RefValue{typeof(^)}, x, ::Base.RefValue{Val{K}}) where {K}
    print(io, "(")
    summary(io, x)
    print(io, ") .^ $K")
end

function _broadcastarray_summary(io::IO, ::typeof(^), x, y)
    print(io, "(")
    summary(io, x)
    print(io, ") .^ ")
    summary(io, y)
end


Base.array_summary(io::IO, C::BroadcastArray, inds::Tuple{Vararg{OneTo}}) = _broadcastarray_summary(io, C)
function Base.array_summary(io::IO, C::BroadcastArray, inds)
    _broadcastarray_summary(io, C)
    print(io, " with indices ", Base.inds2string(inds))
end


function Base.array_summary(io::IO, C::Adjoint{<:Any,<:LazyArray}, inds::Tuple{Vararg{OneTo}})
    print(io, "(")
    summary(io, parent(C))
    print(io, ")'")
end
function Base.array_summary(io::IO, C::Adjoint{<:Any,<:LazyArray}, inds)
    print(io, "(")
    summary(io, parent(C))
    print(io, ")' with indices ", Base.inds2string(inds))
end


function Base.array_summary(io::IO, C::Transpose{<:Any,<:LazyArray}, inds::Tuple{Vararg{OneTo}})
    print(io, "transpose(")
    summary(io, parent(C))
    print(io, ")")
end
function Base.array_summary(io::IO, C::Transpose{<:Any,<:LazyArray}, inds)
    print(io, "transpose(")
    summary(io, parent(C))
    print(io, ") with indices ", Base.inds2string(inds))
end


###
# Mul
###

_broadcast_mul_mul(A, B) = simplify(Mul(broadcast(*, A...), B))
_broadcast_mul_mul(::typeof(*), A, B) = _broadcast_mul_mul(A, B) # maintain back-compatibility with Quasi/ContiuumArrays.jl
_broadcast_mul_simplifiable(op, A, B) = Val(false)
_broadcast_mul_mul(op, A, B) = simplify(Mul(broadcast(op, A...), B))

for op in (:*, :\)
    @eval begin
        _broadcast_mul_simplifiable(::typeof($op), (a,B)::Tuple{Union{AbstractVector,Number},AbstractMatrix}, C) = simplifiable(*, B, C)
        _broadcast_mul_mul(::typeof($op), (a,B)::Tuple{Union{AbstractVector,Number},AbstractMatrix}, C) = broadcast($op, a, (B*C))
    end
end

for op in (:*, :/)
    @eval begin
        _broadcast_mul_simplifiable(::typeof($op), (A,b)::Tuple{AbstractMatrix,Union{AbstractVector,Number}}, C) = simplifiable(*, A, C)
        _broadcast_mul_mul(::typeof($op), (A,b)::Tuple{AbstractMatrix,Union{AbstractVector,Number}}, C) = broadcast($op, (A*C), b)
    end
end



for op in (:*, :/, :\)
    @eval begin
        @inline simplifiable(M::Mul{BroadcastLayout{typeof($op)}}) = _broadcast_mul_simplifiable($op, arguments(BroadcastLayout{typeof($op)}(), M.A), M.B)
        @inline simplifiable(M::Mul{BroadcastLayout{typeof($op)},<:LazyLayouts}) = _broadcast_mul_simplifiable($op, arguments(BroadcastLayout{typeof($op)}(), M.A), M.B)
        @inline simplifiable(M::Mul{BroadcastLayout{typeof($op)},ApplyLayout{typeof(*)}}) = _broadcast_mul_simplifiable($op, arguments(BroadcastLayout{typeof($op)}(), M.A), M.B)
        @inline copy(M::Mul{BroadcastLayout{typeof($op)}}) = _broadcast_mul_mul($op, arguments(BroadcastLayout{typeof($op)}(), M.A), M.B)
        @inline copy(M::Mul{BroadcastLayout{typeof($op)},<:LazyLayouts}) = _broadcast_mul_mul($op, arguments(BroadcastLayout{typeof($op)}(), M.A), M.B)
        @inline copy(M::Mul{BroadcastLayout{typeof($op)},ApplyLayout{typeof(*)}}) = _broadcast_mul_mul($op, arguments(BroadcastLayout{typeof($op)}(), M.A), M.B)
    end
end


for op in (:*, :\, :/)
    @eval begin
        getindex(A::BroadcastMatrix{<:Any,typeof($op),<:Tuple{AbstractVector,AbstractMatrix}}, ::Colon, j::Integer) = broadcast($op, A.args[1], A.args[2][:,j])
        getindex(A::BroadcastMatrix{<:Any,typeof($op),<:Tuple{AbstractMatrix,AbstractVector}}, ::Colon, j::Integer) = broadcast($op, A.args[1][:,j], A.args[2])
    end
end

permutedims(A::BroadcastArray{T}) where T = BroadcastArray{T}(A.f, map(_permutedims,A.args)...)



####
# Dual broadcast: functions of transpose can also behave like transpose
####

@inline broadcastlayout(::Type{F}, ::DualLayout) where F = DualLayout{BroadcastLayout{F}}()


_adjortrans(A::SubArray{<:Any,2, <:Any, <:Tuple{Slice,Any}}) = view(_adjortrans(parent(A)), parentindices(A)[2])
_adjortrans(A::Adjoint) = A'
_adjortrans(A::Transpose) = transpose(A)
