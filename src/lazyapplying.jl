"""
   ApplyStyle

is an abstract type whose subtypes indicate how a lazy function
is materialized. The default is `DefaultApplyStyle` which indicates
that `applied(f, A...)` is materialized as `f(A...)`.
"""
abstract type ApplyStyle end
"""
    AbstractArrayApplyStyle

is an abstract type whose subtypes indicate how a lazy function
is materialized, where the result is an `AbstractArray`.
"""
abstract type AbstractArrayApplyStyle <: ApplyStyle end
"""
    DefaultApplyStyle

indicate that a lazy function application `applied(f, A...)`
is materialized as `f(A...)`.
"""
struct DefaultApplyStyle <: ApplyStyle end

"""
    DefaultArrayApplyStyle

is like DefaultApplyStyle but indicates that the result is an array.
"""
struct DefaultArrayApplyStyle <: AbstractArrayApplyStyle end

@inline ApplyStyle(f, args...) = DefaultApplyStyle()
@inline ApplyStyle(f, ::Type{<:AbstractArray}, args::Type{<:AbstractArray}...) = DefaultArrayApplyStyle()

"""
    Applied(f, A...)

is a lazy version of `f(A...)` that can be manipulated
or materialized in a non-standard manner.
"""
struct Applied{Style, F, Args<:Tuple}
    f::F
    args::Args
end

@inline Applied{Style}(f::F, args::Args) where {Style,F,Args<:Tuple} = Applied{Style,F,Args}(f, args)
@inline Applied{Style}(A::Applied) where Style = Applied{Style}(A.f, A.args)

ndims(a::Applied) = applied_ndims(a.f, a.args...)
eltype(a::Applied) = applied_eltype(a.f, a.args...)
axes(a::Applied) = applied_axes(a.f, a.args...)
size(a::Applied) = applied_size(a.f, a.args...)

call(a) = a.f
call(_, a) = a.f

@inline call(a::AbstractArray) = call(MemoryLayout(a), a)
@inline arguments(a) = a.args #TODO: Deprecate
@inline arguments(::MemoryLayout, a) = a.args #TODO: Deprecate
@inline arguments(f::Function, a) = arguments(ApplyLayout{typeof(f)}(), a)
@inline call(::DualLayout{ML}, a) where ML = call(ML(), a)
@inline arguments(::DualLayout{ML}, a) where ML = arguments(ML(), a)
@inline arguments(a::AbstractArray) = arguments(MemoryLayout(a), a)

@inline check_applied_axes(_...) = nothing

# following repeated due to unexplained allocations
@inline function instantiate(A::Applied{Style}) where Style
    iargs = map(instantiate, A.args)
    check_applied_axes(A.f, iargs...)
    Applied{Style}(A.f, iargs)
end

@inline function applied_instantiate(f, args...)
    iargs = map(instantiate, args)
    check_applied_axes(f, iargs...)
    f, iargs
end

@inline _typesof() = ()
@inline _typesof(a, b...) = tuple(typeof(a), _typesof(b...)...)
@inline _typesof(a, b) = tuple(typeof(a), typeof(b))
@inline _typesof(a, b, c) = tuple(typeof(a), typeof(b), typeof(c))
@inline combine_apply_style(f) = DefaultApplyStyle()
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
# changed in which case it falls back to the default
@inline _default_materialize(A::App, ::App) where App = A.f(A.args...)
@inline _default_materialize(A, _) = materialize(A)
# copy(A::Applied{DefaultApplyStyle}) = A.f(A.args...)
@inline copy(A::Applied) = _default_materialize(materializeargs(A), A)

@inline copyto!(dest, M::Applied) = copyto!(dest, materialize(M))
@inline copyto!(dest::AbstractArray, M::Applied) = copyto!(dest, materialize(M))

@inline broadcastable(M::Applied) = M



similar(M::Applied{<:AbstractArrayApplyStyle}, ::Type{T}, axes) where {T} = Array{T}(undef, length.(axes))
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

@inline matrixfunction_check_applied_axes(a::AbstractMatrix) = axes(a,1) == axes(a,2) || throw(DimensionMismatch("matrix is not square: dimensions are $(axes(a))"))
@inline matrixfunction_check_applied_axes(a...) = nothing

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

@inline ApplyArray{T,N,F,Args}(M::Applied) where {T,N,F,Args} = ApplyArray{T,N,F,Args}(M.f, M.args)
@inline ApplyArray{T,N}(M::Applied{Style,F,Args}) where {T,N,Style,F,Args} = ApplyArray{T,N,F,Args}(instantiate(M))
@inline ApplyArray{T}(M::Applied) where {T} = ApplyArray{T,ndims(M)}(M)
@inline ApplyArray(M::Applied) = ApplyArray{eltype(M)}(M)


@inline ApplyArray(f, factors...) = ApplyArray{applied_eltype(f, factors...)}(f, factors...)
@inline ApplyArray{T}(f, factors...) where T = ApplyArray{T, applied_ndims(f, factors...)}(f, factors...)
@inline function ApplyArray{T,N}(f, factors...) where {T,N}
    f̃, args = applied_instantiate(f, factors...)
    ApplyArray{T,N,typeof(f̃),typeof(args)}(f̃, args)
end

@inline ApplyVector(f, factors...) = ApplyVector{applied_eltype(f, factors...)}(f, factors...)
@inline ApplyMatrix(f, factors...) = ApplyMatrix{applied_eltype(f, factors...)}(f, factors...)

ApplyArray(A::AbstractArray{T,N}) where {T,N} = ApplyArray{T,N}(call(A), arguments(A)...)
ApplyArray{T}(A::AbstractArray{V,N}) where {T,V,N} = ApplyArray{T,N}(call(A), arguments(A)...)
ApplyArray{T,N}(A::AbstractArray{V,N}) where {T,V,N} = ApplyArray{T,N}(call(A), arguments(A)...)
ApplyMatrix(A::AbstractMatrix{T}) where T = ApplyMatrix{T}(call(A), arguments(A)...)
ApplyVector(A::AbstractVector{T}) where T = ApplyVector{T}(call(A), arguments(A)...)

convert(::Type{AbstractArray{T}}, A::ApplyArray{T}) where T = A
convert(::Type{AbstractArray{T}}, A::ApplyArray{<:Any,N}) where {T,N} = ApplyArray{T,N}(A.f, A.args...)
convert(::Type{AbstractArray{T,N}}, A::ApplyArray{T,N}) where {T,N} = A
convert(::Type{AbstractArray{T,N}}, A::ApplyArray{<:Any,N}) where {T,N} = ApplyArray{T,N}(A.f, A.args...)

AbstractArray{T}(A::ApplyArray{T}) where T = copy(A)
AbstractArray{T}(A::ApplyArray{<:Any,N}) where {T,N} = ApplyArray{T,N}(A.f, map(copy,A.args)...)
AbstractArray{T,N}(A::ApplyArray{T,N}) where {T,N} = copy(A)
AbstractArray{T,N}(A::ApplyArray{<:Any,N}) where {T,N} = ApplyArray{T,N}(A.f, map(copy,A.args)...)

@inline axes(A::ApplyArray) = applied_axes(A.f, A.args...)
@inline size(A::ApplyArray) = applied_size(A.f, A.args...)

@inline applied_axes(f, args...) = map(oneto, applied_size(f, args...))



# immutable arrays don't need to copy.
# Some special cases like vcat overload setindex! and therefore
# need to also overload copy
@inline copy(A::ApplyArray{T,N}) where {T,N} = A
map(::typeof(copy), A::ApplyArray) = A


struct LazyArrayApplyStyle <: AbstractArrayApplyStyle end
copy(A::Applied{LazyArrayApplyStyle}) = ApplyArray(A)

@propagate_inbounds getindex(A::ApplyArray{T,N}, kj::Vararg{Integer,N}) where {T,N} = convert(T, Applied(A)[kj...])::T

for F in (:exp, :log, :sqrt, :cos, :sin, :tan, :csc, :sec, :cot,
            :cosh, :sinh, :tanh, :csch, :sech, :coth,
            :acosh, :asinh, :atanh, :acsch, :asech, :acoth,
            :acos, :asin, :atan, :acsc, :asec, :acot)
    @eval begin
        @inline applied_ndims(M::typeof($F), a) = ndims(a)
        @inline applied_axes(::typeof($F), a) = axes(a)
        @inline applied_size(::typeof($F), a) = size(a)
        @inline applied_eltype(::typeof($F), a) = float(eltype(a))
        check_applied_axes(::typeof($F), a...) = matrixfunction_check_applied_axes(a...)
    end
end



###
# show
###


_applyarray_summary(io::IO, C) = _applyarray_summary(io::IO, C.f, arguments(C))
function _applyarray_summary(io::IO, f, args)
    print(io, f)
    print(io, "(")
    if !isempty(args)
        summary(io, first(args))
        for a in tail(args)
            print(io, ", ")
            summary(io, a)
        end
    end
    print(io, ")")
end

Base.array_summary(io::IO, C::ApplyArray, inds::Tuple{Vararg{OneTo}}) = _applyarray_summary(io, C)
function Base.array_summary(io::IO, C::ApplyArray, inds)
    _applyarray_summary(io, C)
    print(io, " with indices ", Base.inds2string(inds))
end

abstract type AbstractLazyLayout <: MemoryLayout end
struct LazyLayout <: AbstractLazyLayout end


MemoryLayout(::Type{<:LazyArray}) = LazyLayout()

transposelayout(::AbstractLazyLayout) = LazyLayout()
conjlayout(::Type{<:Complex}, ::AbstractLazyLayout) = LazyLayout()
sublayout(::AbstractLazyLayout, _) = LazyLayout()
reshapedlayout(::AbstractLazyLayout, _) = LazyLayout()
symmetriclayout(::AbstractLazyLayout) = SymmetricLayout{LazyLayout}()
hermitianlayout(::Type{<:Complex}, ::AbstractLazyLayout) = HermitianLayout{LazyLayout}()
hermitianlayout(::Type{<:Real}, ::AbstractLazyLayout) = SymmetricLayout{LazyLayout}()
triangularlayout(::Type{Tri}, ::AbstractLazyLayout) where Tri = Tri{LazyLayout}()

const LazyLayouts = Union{AbstractLazyLayout, SymmetricLayout{<:AbstractLazyLayout}, HermitianLayout{<:AbstractLazyLayout},
                    TriangularLayout{'L', 'N', <:AbstractLazyLayout}, TriangularLayout{'U', 'N', <:AbstractLazyLayout},
                    TriangularLayout{'L', 'U', <:AbstractLazyLayout}, TriangularLayout{'U', 'U', <:AbstractLazyLayout}}

@inline islazy_layout(::LazyLayouts) = Val(true)
@inline islazy_layout(_) = Val(false)
@inline islazy(A) = islazy_layout(MemoryLayout(A))
                    

struct ApplyLayout{F} <: AbstractLazyLayout end

call(::ApplyLayout{F}, a) where F = F.instance

applylayout(::Type{F}, args...) where F = ApplyLayout{F}()

MemoryLayout(::Type{Applied{Style,F,Args}}) where {Style,F,Args} =
    applylayout(F, tuple_type_memorylayouts(Args)...)
MemoryLayout(::Type{ApplyArray{T,N,F,Args}}) where {T,N,F,Args} =
    applylayout(F, tuple_type_memorylayouts(Args)...)

arguments(::ApplyLayout{F}, A::ApplyArray{<:Any,N,F}) where {N,F} = A.args

@inline Applied(A::AbstractArray) = Applied(call(A), arguments(A)...)

function show(io::IO, A::Applied)
    print(io, "Applied(", A.f)
    for a in A.args
        print(io, ',', a)
    end
    print(io, ')')
end

applybroadcaststyle(::Type{<:AbstractArray{<:Any,N}}, _2) where N = DefaultArrayStyle{N}()
applybroadcaststyle(::Type{<:AbstractArray{<:Any,N}}, ::AbstractLazyLayout) where N = LazyArrayStyle{N}()
BroadcastStyle(M::Type{<:ApplyArray}) = applybroadcaststyle(M, MemoryLayout(M))
BroadcastStyle(M::Type{<:SubArray{<:Any,N,<:ApplyArray}}) where N = applybroadcaststyle(M, MemoryLayout(M))

###
# Eager applications
###

for op in (:+, :-, :*, :\)
    @eval applied(::typeof($op), x::Number, y::Number) = $op(x,y)
end

for op in (:one, :zero)
    @eval applied(::typeof($op), ::Type{T}) where T = $op(T)
end

for f in (view, Base.maybeview, (:))
    @eval applied(::typeof($f), args...) = $f(args...)
end

###
# Lazy getindex
# this uses a lazy-materialize idiom to construct a matrix based
# on the memory layout
###

function copyto!_layout(::LAY, ::LAY, dest::AbstractArray{<:Any,N}, src::AbstractArray{<:Any,N}) where {LAY<:ApplyLayout,N}
    map(copyto!, arguments(dest), arguments(src))
    dest
end

@inline copyto!_layout(_, ::ApplyLayout, dest::AbstractArray, src::AbstractArray) = copyto!(dest, Applied(src))

# avoid infinite-loop
_base_copyto!(dest::AbstractArray{T,N}, src::AbstractArray{T,N}) where {T,N} = Base.invoke(copyto!, NTuple{2,AbstractArray{T,N}}, dest, src)
_base_copyto!(dest::AbstractArray, src::AbstractArray) = Base.invoke(copyto!, NTuple{2,AbstractArray}, dest, src)

##
# triu/tril
##
for tri in (:tril, :triu)
    @eval begin
        applied_axes(::typeof($tri), a, k...) = axes(a)
        applied_size(::typeof($tri), a, k...) = size(a)
        applied_ndims(::typeof($tri), a, k...) = 2
        applied_eltype(::typeof($tri), a, k...) = eltype(a)
        $tri(A::LazyMatrix) = ApplyMatrix($tri, A)
        $tri(A::LazyMatrix, k::Integer) = ApplyMatrix($tri, A, k)
    end
end

getindex(A::ApplyMatrix{T,typeof(triu),<:Tuple{<:AbstractMatrix}}, k::Integer, j::Integer) where T =
    j ≥ k ? A.args[1][k,j] : zero(T)

getindex(A::ApplyMatrix{T,typeof(triu),<:Tuple{<:AbstractMatrix,<:Integer}}, k::Integer, j::Integer) where T =
    j ≥ k+A.args[2] ? A.args[1][k,j] : zero(T)

getindex(A::ApplyMatrix{T,typeof(tril),<:Tuple{<:AbstractMatrix}}, k::Integer, j::Integer) where T =
    j ≤ k ? A.args[1][k,j] : zero(T)

getindex(A::ApplyMatrix{T,typeof(tril),<:Tuple{<:AbstractMatrix,<:Integer}}, k::Integer, j::Integer) where T =
    j ≤ k+A.args[2] ? A.args[1][k,j] : zero(T)


###
# Diagonal
###

# this is needed for infinite diagonal block matrices
copy(D::Diagonal{<:Any,<:LazyArray}) = Diagonal(copy(D.diag))
map(::typeof(copy), D::Diagonal{<:Any,<:LazyArray}) = Diagonal(map(copy,D.diag))
function copy(D::Tridiagonal{<:Any,<:LazyArray})
    if isdefined(D, :du2)
        Tridiagonal(copy(D.dl), copy(D.d), copy(D.du), copy(D.du2))
    else
        Tridiagonal(copy(D.dl), copy(D.d), copy(D.du))
    end
end
function map(::typeof(copy), D::Tridiagonal{<:Any,<:LazyArray})
    if isdefined(D, :du2)
        Tridiagonal(map(copy,D.dl), map(copy,D.d), map(copy,D.du), map(copy,D.du2))
    else
        Tridiagonal(map(copy,D.dl), map(copy,D.d), map(copy,D.du))
    end
end




###
# LazyBandedLayout
###

diagonallayout(::AbstractLazyLayout) = DiagonalLayout{LazyLayout}()
tridiagonallayout(::AbstractLazyLayout, _, _) = TridiagonalLayout{LazyLayout,LazyLayout,LazyLayout}()
tridiagonallayout(_, ::AbstractLazyLayout, _) = TridiagonalLayout{LazyLayout,LazyLayout,LazyLayout}()
tridiagonallayout(_, _, ::AbstractLazyLayout) = TridiagonalLayout{LazyLayout,LazyLayout,LazyLayout}()
tridiagonallayout(::AbstractLazyLayout, _, ::AbstractLazyLayout) = TridiagonalLayout{LazyLayout,LazyLayout,LazyLayout}()
tridiagonallayout(::AbstractLazyLayout, ::AbstractLazyLayout, _) = TridiagonalLayout{LazyLayout,LazyLayout,LazyLayout}()
tridiagonallayout(_, ::AbstractLazyLayout, ::AbstractLazyLayout) = TridiagonalLayout{LazyLayout,LazyLayout,LazyLayout}()
tridiagonallayout(::AbstractLazyLayout, ::AbstractLazyLayout, ::AbstractLazyLayout) = TridiagonalLayout{LazyLayout,LazyLayout,LazyLayout}()
bidiagonallayout(::AbstractLazyLayout, _) = BidiagonalLayout{LazyLayout,LazyLayout}()
bidiagonallayout(_, ::AbstractLazyLayout) = BidiagonalLayout{LazyLayout,LazyLayout}()
bidiagonallayout(::AbstractLazyLayout, ::AbstractLazyLayout) = BidiagonalLayout{LazyLayout,LazyLayout}()
symtridiagonallayout(::AbstractLazyLayout, _) = SymTridiagonalLayout{LazyLayout,LazyLayout}()
symtridiagonallayout(_, ::AbstractLazyLayout) = SymTridiagonalLayout{LazyLayout,LazyLayout}()
symtridiagonallayout(::AbstractLazyLayout, ::AbstractLazyLayout) = SymTridiagonalLayout{LazyLayout,LazyLayout}()

