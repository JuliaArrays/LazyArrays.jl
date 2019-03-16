

abstract type ApplyStyle end
abstract type AbstractArrayApplyStyle <: ApplyStyle end
struct DefaultApplyStyle <: ApplyStyle end
struct LayoutApplyStyle{Layouts<:Tuple} <: AbstractArrayApplyStyle
    layouts::Layouts
end

ApplyStyle(f, args...) = DefaultApplyStyle()

struct Applied{Style, F, Args<:Tuple}
    style::Style
    f::F
    args::Args
end

Applied(f, args...) = Applied(ApplyStyle(f, args...), f, args)
applied(f, args...) = Applied(f, args...)

materialize(A::Applied) = _default_materialize(A)
materializeargs(A::Applied) = applied(A.f, materialize.(A.args)...)

# the following materialzes the args and calls materialize again, unless it hasn't
# changed in which case it falls back to the default
__default_materialize(A::App, ::App) where App = A.f(A.args...)
__default_materialize(A, _) where App = materialize(A)
_default_materialize(A::Applied) = __default_materialize(materializeargs(A), A)


# _materialize is for applied with axes, which defaults to using copyto!
materialize(M::Applied{<:AbstractArrayApplyStyle}) = _materialize(M, axes(M))
_materialize(A::Applied, _) = _default_materialize(A)

@inline copyto!(dest::AbstractArray, M::Applied{DefaultApplyStyle}) = copyto!(dest, materialize(M))
@inline copyto!(dest, M::Applied{DefaultApplyStyle}) = copyto!(dest, materialize(M))

@inline copyto!(dest::AbstractArray, M::Applied{<:LayoutApplyStyle}) =
    _copyto!(MemoryLayout(dest), dest, materializeargs(M))

# Used for when a lazy version should be constructed on materialize
struct LazyArrayApplyStyle <: AbstractArrayApplyStyle end
broadcastable(M::Applied) = M



similar(M::Applied{<:AbstractArrayApplyStyle}, ::Type{T}, ::NTuple{N,OneTo{Int}}) where {T,N} = Array{T}(undef, size(M))
similar(M::Applied{<:AbstractArrayApplyStyle}, ::Type{T}) where T = similar(M, T, axes(M))

similar(M::Applied) = similar(M, eltype(M))

struct ApplyBroadcastStyle <: BroadcastStyle end

@inline copyto!(dest::AbstractArray, bc::Broadcasted{ApplyBroadcastStyle}) =
    copyto!(dest, first(bc.args))
# Use default broacasting in general
@inline _copyto!(_, dest, bc::Broadcasted) = copyto!(dest, Broadcasted{Nothing}(bc.f, bc.args, bc.axes))

BroadcastStyle(::Type{<:Applied}) = ApplyBroadcastStyle()


struct MatrixFunctionStyle{F} <: AbstractArrayApplyStyle end

for f in (:exp, :sin, :cos, :sqrt)
    @eval ApplyStyle(::typeof($f), ::AbstractMatrix) = MatrixFunctionStyle{typeof($f)}()
end

materialize(A::Applied{<:MatrixFunctionStyle,<:Any,<:Tuple{<:Any}}) =
    _default_materialize(A)

axes(A::Applied{<:MatrixFunctionStyle}) = axes(first(A.args))
size(A::Applied{<:MatrixFunctionStyle}) = size(first(A.args))
eltype(A::Applied{<:MatrixFunctionStyle}) = eltype(first(A.args))

getindex(A::Applied{<:MatrixFunctionStyle}, k::Int, j::Int) =
    materialize(A)[k,j]


struct ApplyArray{T, N, App<:Applied} <: AbstractArray{T,N}
    applied::App
end

const ApplyVector{T, App<:Applied} = ApplyArray{T, 1, App}
const ApplyMatrix{T, App<:Applied} = ApplyArray{T, 2, App}


ApplyArray{T,N}(M::App) where {T,N,App<:Applied} = ApplyArray{T,N,App}(M)
ApplyArray{T}(M::Applied) where {T} = ApplyArray{T,ndims(M)}(M)
ApplyArray(M::Applied) = ApplyArray{eltype(M)}(M)
ApplyVector(M::Applied) = ApplyVector{eltype(M)}(M)
ApplyMatrix(M::Applied) = ApplyMatrix{eltype(M)}(M)

ApplyArray(f, factors...) = ApplyArray(applied(f, factors...))
ApplyArray{T}(f, factors...) where T = ApplyArray{T}(applied(f, factors...))
ApplyArray{T,N}(f, factors...) where {T,N} = ApplyArray{T,N}(applied(f, factors...))

ApplyVector(f, factors...) = ApplyVector(applied(f, factors...))
ApplyMatrix(f, factors...) = ApplyMatrix(applied(f, factors...))

axes(A::ApplyArray) = axes(A.applied)
size(A::ApplyArray) = map(length, axes(A))

IndexStyle(::ApplyArray{<:Any,1}) = IndexLinear()

@propagate_inbounds getindex(A::ApplyArray{T,N}, kj::Vararg{Int,N}) where {T,N} =
    materialize(A.applied)[kj...]

materialize(A::Applied{LazyArrayApplyStyle}) = ApplyArray(A)


# adjoint(A::MulArray) = MulArray(reverse(adjoint.(A.applied.args))...)
# transpose(A::MulArray) = MulArray(reverse(transpose.(A.applied.args))...)


struct  ApplyLayout{F, LAY} <: MemoryLayout
    f::F
    layouts::LAY
end

MemoryLayout(M::ApplyArray) = ApplyLayout(M.applied.f, MemoryLayout.(M.applied.args))

function show(io::IO, A::Applied) 
    print(io, "Applied(", A.f)
    for a in A.args
        print(io, ',', a)
    end
    print(io, ')')
end


# _flatten(A::ApplyArray, B...) = _flatten(A.applied.args..., B...)
# flatten(A::MulArray) = MulArray(Mul(_flatten(A.applied.args...)))
