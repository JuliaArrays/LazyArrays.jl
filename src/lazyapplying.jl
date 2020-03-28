

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


call(a) = a.f
call(_, a) = a.f
call(LAY, a::SubArray) = call(LAY, parent(a))
call(a::AbstractArray) = call(MemoryLayout(typeof(a)), a)
arguments(a) = a.args
arguments(_, a) = a.args
arguments(a::AbstractArray) = arguments(MemoryLayout(typeof(a)), a)

@inline check_applied_axes(A::Applied) = nothing

@inline function instantiate(A::Applied{Style}) where Style
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

@inline materialize(A::Applied) = copy(instantiate(A))
@inline materializeargs(A::Applied) = applied(A.f, materialize.(A.args)...)

# the following materialzes the args and calls materialize again, unless it hasn't
# changed in which case it falls back to the default
@inline _default_materialize(A::App, ::App) where App = A.f(A.args...)
@inline _default_materialize(A, _) = materialize(A)
# copy(A::Applied{DefaultApplyStyle}) = A.f(A.args...)
@inline copy(A::Applied) = _default_materialize(materializeargs(A), A)

@inline copyto!(dest, M::Applied) = copyto!(dest, materialize(M))
@inline copyto!(dest::AbstractArray, M::Applied) = copyto!(dest, materialize(M))

@inline broadcastable(M::Applied) = M



similar(M::Applied{<:AbstractArrayApplyStyle}, ::Type{T}, axes) where {T,N} = Array{T}(undef, length.(axes))
similar(M::Applied{<:AbstractArrayApplyStyle}, ::Type{T}) where T = similar(M, T, axes(M))
similar(M::Applied) = similar(M, eltype(M))

@inline axes(A::Applied, j) = axes(A)[j]

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


@inline function copyto!(dest::AbstractArray, bc::Broadcasted{ApplyArrayBroadcastStyle{N}}) where N 
    if length(bc.args) == 1
        copyto!(dest, first(bc.args))
        if bc.f !== identity
            dest .= bc.f.(dest)
        end
    else
        bc′ = mapbc(bc) do x
            if x isa Applied
                materialize(x)
            else
                x
            end
        end
        materialize!(dest, bc′)
    end
    return dest
end

# Map over all nested Broadcasted and their arguments.  Using `broadcasted`
# instead of `Broadcasted` to re-process arguments via `broadcastable`.
@inline mapbc(f, bc::Broadcasted) =
    f(broadcasted(bc.f, map(a -> mapbc(f, a), bc.args)...))
@inline mapbc(f, x) = f(x)

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
abstract type LazyArray{T,N} <: LayoutArray{T,N} end

const LazyMatrix{T} = LazyArray{T,2}
const LazyVector{T} = LazyArray{T,1}

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

@inline axes(A::ApplyArray) = axes(Applied(A))
@inline size(A::ApplyArray) = map(length, axes(A))
@inline copy(A::ApplyArray{T,N}) where {T,N} = ApplyArray{T,N}(A.f, map(copy,A.args)...)


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

abstract type AbstractLazyLayout <: MemoryLayout end
struct LazyLayout <: AbstractLazyLayout end


MemoryLayout(::Type{<:LazyArray}) = LazyLayout()

transposelayout(L::LazyLayout) = L
conjlayout(L::LazyLayout) = L
sublayout(L::LazyLayout, _) = L
reshapedlayout(::LazyLayout, _) = LazyLayout()

combine_mul_styles(::LazyLayout) = LazyArrayApplyStyle()
result_mul_style(::LazyArrayApplyStyle, ::LazyArrayApplyStyle) = LazyArrayApplyStyle()
result_mul_style(::LazyArrayApplyStyle, _) = LazyArrayApplyStyle()
result_mul_style(_, ::LazyArrayApplyStyle) = LazyArrayApplyStyle()


struct  ApplyLayout{F} <: MemoryLayout end

applylayout(::Type{F}, args...) where F = ApplyLayout{F}()

MemoryLayout(::Type{Applied{Style,F,Args}}) where {Style,F,Args} = 
    applylayout(F, tuple_type_memorylayouts(Args)...)
MemoryLayout(::Type{ApplyArray{T,N,F,Args}}) where {T,N,F,Args} = 
    applylayout(F, tuple_type_memorylayouts(Args)...)

@inline Applied(A::AbstractArray) = Applied(call(A), arguments(A)...)
@inline ApplyArray(A::AbstractArray) = ApplyArray(call(A), arguments(A)...)

function show(io::IO, A::Applied) 
    print(io, "Applied(", A.f)
    for a in A.args
        print(io, ',', a)
    end
    print(io, ')')
end

applybroadcaststyle(::Type{<:AbstractArray{<:Any,N}}, _2) where N = DefaultArrayStyle{N}()
applybroadcaststyle(::Type{<:AbstractArray{<:Any,N}}, ::LazyLayout) where N = LazyArrayStyle{N}()
BroadcastStyle(M::Type{<:ApplyArray}) = applybroadcaststyle(M, MemoryLayout(M))

replace_in_print_matrix(A::LazyMatrix, i::Integer, j::Integer, s::AbstractString) =
    i in colsupport(A,j) ? s : replace_with_centered_mark(s)

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

function _copyto!(::LAY, ::LAY, dest::AbstractArray{T,N}, src::AbstractArray{T,N}) where {LAY<:ApplyLayout,T,N} 
    map(copyto!, arguments(dest), arguments(src))
    dest
end

@inline _copyto!(_, ::ApplyLayout, dest::AbstractArray{T,N}, src::AbstractArray{T,N}) where {T,N} = 
    copyto!(dest, Applied(src))
@inline _copyto!(_, ::ApplyLayout, dest::AbstractArray, src::AbstractArray) = copyto!(dest, Applied(src))    

# avoid infinite-loop
_base_copyto!(dest::AbstractArray{T,N}, src::AbstractArray{T,N}) where {T,N} = Base.invoke(copyto!, NTuple{2,AbstractArray{T,N}}, dest, src)
_base_copyto!(dest::AbstractArray, src::AbstractArray) = Base.invoke(copyto!, NTuple{2,AbstractArray}, dest, src)
@inline copyto!(dest::AbstractArray, M::Applied{LazyArrayApplyStyle}) = _base_copyto!(dest, materialize(M))

## 
# triu/tril
##
for tri in (:tril, :triu)
    for op in (:axes, :size)
        @eval begin
            $op(A::Applied{<:Any,typeof($tri)}) = $op(first(A.args))
            $op(A::Applied{<:Any,typeof($tri)}, j) = $op(first(A.args), j)
        end
    end
    @eval begin 
        ndims(::Applied{<:Any,typeof($tri)}) = 2
        eltype(A::Applied{<:Any,typeof($tri)}) = eltype(first(A.args))
        $tri(A::LazyMatrix) = ApplyMatrix($tri, A)
        $tri(A::LazyMatrix, k::Integer) = ApplyMatrix($tri, A, k)
    end
end

getindex(A::ApplyMatrix{T,typeof(triu),<:Tuple{<:AbstractMatrix}}, k::Integer, j::Integer) where T = 
    j ≥ k ? A.args[1][k,j] : zero(T)

getindex(A::ApplyMatrix{T,typeof(triu),<:Tuple{<:AbstractMatrix,<:Integer}}, k::Integer, j::Integer) where T = 
    j ≥ k+A.args[2] ? A.args[1][k,j] : zero(T)    

getindex(A::ApplyMatrix{T,typeof(tril),<:Tuple{<:AbstractMatrix}}, k::Integer, j::Integer) where T = 
    j ≤ k ? A.args[1][k,j] : zero(T)

getindex(A::ApplyMatrix{T,typeof(tril),<:Tuple{<:AbstractMatrix,<:Integer}}, k::Integer, j::Integer) where T = 
    j ≤ k+A.args[2] ? A.args[1][k,j] : zero(T)    


replace_in_print_matrix(A::ApplyMatrix{<:Any,typeof(triu),<:Tuple{<:AbstractMatrix}}, i::Integer, j::Integer, s::AbstractString) =
    j ≥ i ? replace_in_print_matrix(A.args[1], i, j, s) : replace_with_centered_mark(s)
replace_in_print_matrix(A::ApplyMatrix{<:Any,typeof(triu),<:Tuple{<:AbstractMatrix,<:Integer}}, i::Integer, j::Integer, s::AbstractString) =
    j ≥ i+A.args[2] ? replace_in_print_matrix(A.args[1], i, j, s) : replace_with_centered_mark(s)    


replace_in_print_matrix(A::ApplyMatrix{<:Any,typeof(tril),<:Tuple{<:AbstractMatrix}}, i::Integer, j::Integer, s::AbstractString) =
    j ≤ i ? replace_in_print_matrix(A.args[1], i, j, s) : replace_with_centered_mark(s)
replace_in_print_matrix(A::ApplyMatrix{<:Any,typeof(tril),<:Tuple{<:AbstractMatrix,<:Integer}}, i::Integer, j::Integer, s::AbstractString) =
    j ≤ i+A.args[2] ? replace_in_print_matrix(A.args[1], i, j, s) : replace_with_centered_mark(s)    
