



abstract type ApplyStyle end
struct DefaultApplyStyle <: ApplyStyle end
struct LayoutApplyStyle{Layouts<:Tuple} <: ApplyStyle
    layouts::Layouts
end

ApplyStyle(f, args...) = DefaultApplyStyle()

struct Applied{Style<:ApplyStyle, F, Args<:Tuple}
    style::Style
    f::F
    args::Args
end

applied(f, args...) = Applied(ApplyStyle(f, args...), f, args)
materialize(A::Applied{DefaultApplyStyle,<:Any,<:Tuple{<:Any}}) =
    A.f(materialize(first(A.args)))
materialize(A::Applied{DefaultApplyStyle}) = A.f(materialize.(A.args)...)



similar(M::Applied) = similar(M, eltype(M))


struct MatrixFunctionStyle{F} <: ApplyStyle end

for f in (:exp, :sin, :cos, :sqrt)
    @eval ApplyStyle(::typeof($f), ::AbstractMatrix) = MatrixFunctionStyle{typeof($f)}()
end

materialize(A::Applied{<:MatrixFunctionStyle,<:Any,<:Tuple{<:Any}}) =
    A.f(materialize(first(A.args)))

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

@inline copyto!(dest::AbstractArray, M::Applied) = _copyto!(MemoryLayout(dest), dest, M)
@inline _copyto!(_, dest::AbstractArray, M::Applied) = copyto!(dest, materialize(M))

broadcastable(M::Applied) = M

# adjoint(A::MulArray) = MulArray(reverse(adjoint.(A.mul.args))...)
# transpose(A::MulArray) = MulArray(reverse(transpose.(A.mul.args))...)


struct  ApplyLayout{F, LAY} <: MemoryLayout
    f::F
    layouts::LAY
end

MemoryLayout(M::ApplyArray) = ApplyLayout(M.applied.f, MemoryLayout.(M.applied.args))


# _flatten(A::ApplyArray, B...) = _flatten(A.mul.args..., B...)
# flatten(A::MulArray) = MulArray(Mul(_flatten(A.mul.args...)))



const Add{Factors<:Tuple} = Applied{<:Any, typeof(+), Factors}

size(M::Add, p::Int) = size(M)[p]
axes(M::Add, p::Int) = axes(M)[p]
ndims(M::Add) = ndims(first(M.args))

length(M::Add) = prod(size(M))
size(M::Add) = length.(axes(M))
axes(M::Add) = axes(first(M.args))


eltype(M::Add) = Base._return_type(+, eltype.(M.args))

const AddArray{T,N,Factors<:Tuple} = ApplyArray{T,N,<:Add{Factors}}
const AddVector{T,Factors<:Tuple} = AddArray{T,1,Factors}
const AddMatrix{T,Factors<:Tuple} = AddArray{T,2,Factors}

AddArray(factors...) = ApplyArray(+, factors...)

"""
    Add(A1, A2, …, AN)

A lazy representation of `A1 + A2 + … + AN`; i.e., a shorthand for `applied(+, A1, A2, …, AN)`.
"""
Add(As...) = applied(+, As...)
