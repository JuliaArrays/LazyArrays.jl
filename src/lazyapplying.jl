

abstract type ApplyStyle end
abstract type AbstractArrayApplyStyle <: ApplyStyle end
struct DefaultApplyStyle <: ApplyStyle end
struct DefaultArrayApplyStyle <: AbstractArrayApplyStyle end

ApplyStyle(f, args::Type...) = DefaultApplyStyle()
ApplyStyle(f, ::Type{<:AbstractArray}, args::Type{<:AbstractArray}...) = DefaultArrayApplyStyle()

struct Applied{Style, F, Args<:Tuple}
    f::F
    args::Args
end

Applied{Style}(f::F, args::Args) where {Style,F,Args<:Tuple} = 
    Applied{Style,Core.Typeof(f),Args}(f, args)

check_applied_axes(A::Applied) = nothing

function instantiate(A::Applied) 
    check_applied_axes(A)
    A
end

_typesof() = ()
_typesof(a, b...) = tuple(typeof(a), _typesof(b...)...)
_typesof(a, b) = tuple(typeof(a), typeof(b))
_typesof(a, b, c) = tuple(typeof(a), typeof(b), typeof(c))
combine_apply_style(f, a...) = ApplyStyle(f, _typesof(a...)...)
combine_apply_style(f, a, b) = ApplyStyle(f, typeof(a), typeof(b))
combine_apply_style(f, a, b, c) = ApplyStyle(f, typeof(a), typeof(b), typeof(c))


Applied(f, args...) = Applied{typeof(combine_apply_style(f, args...))}(f, args)
applied(f, args...) = Applied(f, args...)
apply(f, args...) = materialize(applied(f, args...))
apply!(f, args...) = materialize!(applied(f, args...))

materialize(A::Applied) = copy(instantiate(A))
materializeargs(A::Applied) = applied(A.f, materialize.(A.args)...)

# the following materialzes the args and calls materialize again, unless it hasn't
# changed in which case it falls back to the default
__default_materialize(A::App, ::App) where App = A.f(A.args...)
__default_materialize(A, _) where App = materialize(A)
copy(A::Applied) = __default_materialize(materializeargs(A), A)


# _materialize is for applied with axes, which defaults to using copyto!
materialize(M::Applied{<:AbstractArrayApplyStyle}) = _materialize(instantiate(M), axes(M))
_materialize(A::Applied{<:AbstractArrayApplyStyle}, _) = copy(A)
_materialize(A::Applied, _) = copy(A)

@inline copyto!(dest, M::Applied) = copyto!(dest, materialize(M))
@inline copyto!(dest::AbstractArray, M::Applied) = copyto!(dest, materialize(M))

broadcastable(M::Applied) = M



similar(M::Applied{<:AbstractArrayApplyStyle}, ::Type{T}, axes) where {T,N} = Array{T}(undef, length.(axes))
similar(M::Applied{<:AbstractArrayApplyStyle}, ::Type{T}) where T = similar(M, T, axes(M))
similar(M::Applied) = similar(M, eltype(M))

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

axes(A::Applied{<:MatrixFunctionStyle}) = axes(first(A.args))
size(A::Applied{<:MatrixFunctionStyle}) = size(first(A.args))
eltype(A::Applied{<:MatrixFunctionStyle}) = eltype(first(A.args))

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

function getproperty(A::ApplyArray, d::Symbol)
    if d == :applied
        applied(A.f, A.args...)
    else
        getfield(A, d)
    end
end

axes(A::ApplyArray) = axes(A.applied)
size(A::ApplyArray) = map(length, axes(A))
copy(A::ApplyArray) = copy(A.applied)


struct LazyArrayApplyStyle <: AbstractArrayApplyStyle end
copy(A::Applied{<:LazyArrayApplyStyle}) = ApplyArray(A)

IndexStyle(::ApplyArray{<:Any,1}) = IndexLinear()

@propagate_inbounds getindex(A::ApplyArray{T,N}, kj::Vararg{Int,N}) where {T,N} = A.applied[kj...]


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
copy(A::Applied{LazyArrayApplyStyle}) = ApplyArray(A)


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

