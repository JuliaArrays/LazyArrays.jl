

abstract type ApplyStyle end
abstract type AbstractArrayApplyStyle <: ApplyStyle end
struct DefaultApplyStyle <: ApplyStyle end
struct DefaultArrayApplyStyle <: AbstractArrayApplyStyle end

@inline ApplyStyle(f, args...) = DefaultApplyStyle()
@inline ApplyStyle(f, ::Type{<:AbstractArray}, args::Type{<:AbstractArray}...) = DefaultArrayApplyStyle()

struct Applied{Style, F, Args<:Tuple}
    f::F
    args::Args
end

@inline Applied{Style}(f::F, args::Args) where {Style,F,Args<:Tuple} = Applied{Style,F,Args}(f, args)
@inline Applied{Style}(A::Applied) where Style = Applied{Style}(A.f, A.args)

@inline check_applied_axes(A::Applied) = nothing

function instantiate(A::Applied{Style}) where Style
    check_applied_axes(A)
    Applied{Style}(A.f, map(instantiate, A.args))
end

@inline _typesof() = ()
@inline _typesof(a, b...) = tuple(typeof(a), _typesof(b...)...)
@inline _typesof(a, b) = tuple(typeof(a), typeof(b))
@inline _typesof(a, b, c) = tuple(typeof(a), typeof(b), typeof(c))
@inline combine_apply_style(f, a...) = ApplyStyle(f, _typesof(a...)...)
@inline combine_apply_style(f, a, b) = ApplyStyle(f, typeof(a), typeof(b))
@inline combine_apply_style(f, a, b, c) = ApplyStyle(f, typeof(a), typeof(b), typeof(c))


@inline Applied(f, args...) = Applied{typeof(combine_apply_style(f, args...))}(f, args)
@inline applied(f, args...) = Applied(f, args...)
@inline apply(f, args...) = materialize(applied(f, args...))
@inline apply!(f, args...) = materialize!(applied(f, args...))

materialize(A::Applied) = copy(instantiate(A))
materializeargs(A::Applied) = applied(A.f, materialize.(A.args)...)

# the following materialzes the args and calls materialize again, unless it hasn't
# changed in which case it falls back to the default
_default_materialize(A::App, ::App) where App = A.f(A.args...)
_default_materialize(A, _) = materialize(A)
# copy(A::Applied{DefaultApplyStyle}) = A.f(A.args...)
copy(A::Applied) = _default_materialize(materializeargs(A), A)

@inline copyto!(dest, M::Applied) = copyto!(dest, materialize(M))
@inline copyto!(dest::AbstractArray, M::Applied) = copyto!(dest, materialize(M))

broadcastable(M::Applied) = M



similar(M::Applied{<:AbstractArrayApplyStyle}, ::Type{T}, axes) where {T,N} = Array{T}(undef, length.(axes))
similar(M::Applied{<:AbstractArrayApplyStyle}, ::Type{T}) where T = similar(M, T, axes(M))
similar(M::Applied) = similar(M, eltype(M))

axes(A::Applied, j) = axes(A)[j]

struct ApplyBroadcastStyle <: BroadcastStyle end
struct ApplyArrayBroadcastStyle{N} <: Broadcast.AbstractArrayStyle{N} end
ApplyArrayBroadcastStyle{N}(::Val{N}) where N = ApplyArrayBroadcastStyle{N}()


BroadcastStyle(::Type{<:Applied}) = ApplyBroadcastStyle()
# temporary fix
BroadcastStyle(::Type{<:Applied{<:Any,typeof(*),<:Tuple{<:AbstractMatrix,<:AbstractVector}}}) = ApplyArrayBroadcastStyle{1}()
BroadcastStyle(::Type{<:Applied{<:Any,typeof(*),<:Tuple{<:AbstractMatrix,<:AbstractMatrix}}}) = ApplyArrayBroadcastStyle{2}()
BroadcastStyle(::Type{<:Applied{<:Any,typeof(*),<:Tuple{<:AbstractVector,<:AbstractMatrix}}}) = ApplyArrayBroadcastStyle{2}()

BroadcastStyle(::ApplyArrayBroadcastStyle{Any}, b::DefaultArrayStyle) = b
BroadcastStyle(::ApplyArrayBroadcastStyle{N}, b::DefaultArrayStyle{N}) where N = b
BroadcastStyle(::ApplyArrayBroadcastStyle{M}, b::DefaultArrayStyle{N}) where {M,N} =
    typeof(b)(Val(max(M, N)))

similar(bc::Broadcasted{ApplyArrayBroadcastStyle{N}}, ::Type{ElType}) where {N,ElType} =
    similar(Array{ElType}, axes(bc))

@inline function copyto!(dest::AbstractArray, bc::Broadcasted{ApplyBroadcastStyle}) 
    @assert length(bc.args) == 1
    copyto!(dest, first(bc.args))
end
@inline function copyto!(dest::AbstractArray, bc::Broadcasted{ApplyArrayBroadcastStyle{N}}) where N 
    @assert length(bc.args) == 1
    copyto!(dest, first(bc.args))    
end

struct MatrixFunctionStyle{F} <: AbstractArrayApplyStyle end

for f in (:exp, :sin, :cos, :sqrt)
    @eval ApplyStyle(::typeof($f), ::Type{<:AbstractMatrix}) = MatrixFunctionStyle{typeof($f)}()
end

function check_applied_axes(A::Applied{<:MatrixFunctionStyle}) 
    length(A.args) == 1 || throw(ArgumentError("MatrixFunctions only defined with 1 arg"))
    axes(A.args[1],1) == axes(A.args[1],2) || throw(DimensionMismatch("matrix is not square: dimensions are $axes(A.args[1])"))
end

for op in (:axes, :size)
    @eval begin
        $op(A::Applied{<:MatrixFunctionStyle}) = $op(first(A.args))
        $op(A::Applied{<:MatrixFunctionStyle}, j) = $op(first(A.args), j)
    end
end

ndims(A::Applied{<:MatrixFunctionStyle}) = ndims(first(A.args))

eltype(A::Applied{<:MatrixFunctionStyle}) = float(eltype(first(A.args)))

getindex(A::Applied, kj...) = materialize(A)[kj...]

"""
    LazyArray(x::Applied) :: ApplyArray
    LazyArray(x::Broadcasted) :: BroadcastArray

Wrap a lazy object that wraps a computation producing an array to an
array.
"""
abstract type LazyArray{T,N} <: AbstractArray{T,N} end

struct ApplyArray{T, N, F, Args<:Tuple} <: LazyArray{T,N}
    f::F
    args::Args
end

const ApplyVector{T, F, Args<:Tuple} = ApplyArray{T, 1, F, Args}
const ApplyMatrix{T, F, Args<:Tuple} = ApplyArray{T, 2, F, Args}

LazyArray(A::Applied) = ApplyArray(A)

ApplyArray{T,N,F,Args}(M::Applied) where {T,N,F,Args} = ApplyArray{T,N,F,Args}(M.f, M.args)
ApplyArray{T,N}(M::Applied{Style,F,Args}) where {T,N,Style,F,Args} = ApplyArray{T,N,F,Args}(instantiate(M))
ApplyArray{T}(M::Applied) where {T} = ApplyArray{T,ndims(M)}(M)
ApplyArray(M::Applied) = ApplyArray{eltype(M)}(M)
ApplyVector(M::Applied) = ApplyVector{eltype(M)}(M)
ApplyMatrix(M::Applied) = ApplyMatrix{eltype(M)}(M)

ApplyArray(f, factors...) = ApplyArray(applied(f, factors...))
ApplyArray{T}(f, factors...) where T = ApplyArray{T}(applied(f, factors...))
ApplyArray{T,N}(f, factors...) where {T,N} = ApplyArray{T,N}(applied(f, factors...))

ApplyVector(f, factors...) = ApplyVector(applied(f, factors...))
ApplyMatrix(f, factors...) = ApplyMatrix(applied(f, factors...))

convert(::Type{AbstractArray{T}}, A::ApplyArray{T}) where T = A
convert(::Type{AbstractArray{T}}, A::ApplyArray{<:Any,N}) where {T,N} = ApplyArray{T,N}(A.f, A.args...)
convert(::Type{AbstractArray{T,N}}, A::ApplyArray{T,N}) where {T,N} = A
convert(::Type{AbstractArray{T,N}}, A::ApplyArray{<:Any,N}) where {T,N} = ApplyArray{T,N}(A.f, A.args...)

AbstractArray{T}(A::ApplyArray{T}) where T = copy(A)
AbstractArray{T}(A::ApplyArray{<:Any,N}) where {T,N} = ApplyArray{T,N}(A.f, map(copy,A.args)...)
AbstractArray{T,N}(A::ApplyArray{T,N}) where {T,N} = copy(A)
AbstractArray{T,N}(A::ApplyArray{<:Any,N}) where {T,N} = ApplyArray{T,N}(A.f, map(copy,A.args)...)


@inline Applied(A::ApplyArray) = applied(A.f, A.args...)
@inline axes(A::ApplyArray) = axes(Applied(A))
@inline size(A::ApplyArray) = map(length, axes(A))
@inline copy(A::ApplyArray) = ApplyArray(A.f, map(copy,A.args)...)


struct LazyArrayApplyStyle <: AbstractArrayApplyStyle end
copy(A::Applied{LazyArrayApplyStyle}) = ApplyArray(A)

@propagate_inbounds getindex(A::ApplyArray{T,N}, kj::Vararg{Integer,N}) where {T,N} = convert(T, Applied(A)[kj...])::T
@propagate_inbounds getindex(A::Applied{LazyArrayApplyStyle}, kj...) = materialize(Applied{DefaultArrayApplyStyle}(A))[kj...]

for F in (:exp, :log, :sqrt, :cos, :sin, :tan, :csc, :sec, :cot,
            :cosh, :sinh, :tanh, :csch, :sech, :coth,
            :acosh, :asinh, :atanh, :acsch, :asech, :acoth,
            :acos, :asin, :atan, :acsc, :asec, :acot)
    @eval begin
        ndims(M::Applied{LazyArrayApplyStyle,typeof($F)}) = ndims(first(M.args))
        axes(M::Applied{LazyArrayApplyStyle,typeof($F)}) = axes(first(M.args))
        size(M::Applied{LazyArrayApplyStyle,typeof($F)}) = size(first(M.args))
        eltype(M::Applied{LazyArrayApplyStyle,typeof($F)}) = eltype(first(M.args))
    end
end

struct LazyLayout <: MemoryLayout end


MemoryLayout(::Type{<:LazyArray}) = LazyLayout()

transposelayout(::LazyLayout) = LazyLayout()
conjlayout(::LazyLayout) = LazyLayout()

combine_mul_styles(::LazyLayout) = LazyArrayApplyStyle()
result_mul_style(::LazyArrayApplyStyle, ::LazyArrayApplyStyle) = LazyArrayApplyStyle()
result_mul_style(::LazyArrayApplyStyle, _) = LazyArrayApplyStyle()
result_mul_style(_, ::LazyArrayApplyStyle) = LazyArrayApplyStyle()


struct  ApplyLayout{F, LAY} <: MemoryLayout end

MemoryLayout(M::Type{Applied{Style,F,Args}}) where {Style,F,Args} = ApplyLayout{F,tuple_type_memorylayouts(Args)}()
MemoryLayout(M::Type{ApplyArray{T,N,F,Args}}) where {T,N,F,Args} = ApplyLayout{F,tuple_type_memorylayouts(Args)}()

function show(io::IO, A::Applied) 
    print(io, "Applied(", A.f)
    for a in A.args
        print(io, ',', a)
    end
    print(io, ')')
end

applybroadcaststyle(_1, _2) = DefaultArrayStyle{2}()
BroadcastStyle(M::Type{<:ApplyArray}) = applybroadcaststyle(M, MemoryLayout(M))

Base.replace_in_print_matrix(A::ApplyMatrix, i::Integer, j::Integer, s::AbstractString) =
    i in colsupport(A,j) ? s : Base.replace_with_centered_mark(s)

### 
# Number special cases
###

for op in (:+, :-, :*, :\)
    @eval applied(::typeof($op), x::Number, y::Number) = $op(x,y)
end

for op in (:one, :zero)
    @eval applied(::typeof($op), ::Type{T}) where T = $op(T)
end

###
# Lazy getindex
# this uses a lazy-materialize idiom to construct a matrix based
# on the memory layout
###

@inline sub_materialize(_, V) = Array(V)
@inline sub_materialize(V::SubArray) = sub_materialize(MemoryLayout(typeof(V)), V)

@inline lazy_getindex(A, I...) = sub_materialize(view(A, I...))


@inline getindex(A::ApplyMatrix, kr::Colon, jr::Colon) = lazy_getindex(A, kr, jr)
@inline getindex(A::ApplyMatrix, kr::Colon, jr::AbstractUnitRange) = lazy_getindex(A, kr, jr)
@inline getindex(A::ApplyMatrix, kr::AbstractUnitRange, jr::Colon) = lazy_getindex(A, kr, jr)
@inline getindex(A::ApplyMatrix, kr::AbstractUnitRange, jr::AbstractUnitRange) = lazy_getindex(A, kr, jr)


diagonallayout(::LazyLayout) = LazyLayout()
diagonallayout(::ApplyLayout) = DiagonalLayout{LazyLayout}()