"""
    MulAddStyle

indicates that an `Applied` object should be materialised
via `ArrayLayouts.MulAdd`.
"""
struct MulAddStyle <: AbstractArrayApplyStyle end


ApplyStyle(::typeof(*), ::Type{α}, ::Type{A}, ::Type{B}) where {α<:Number, A<:AbstractMatrix,B<:AbstractVector} =
    MulAddStyle()
ApplyStyle(::typeof(*), ::Type{α}, ::Type{A}, ::Type{B}) where {α<:Number, A<:AbstractMatrix,B<:AbstractMatrix} =
    MulAddStyle()
ApplyStyle(::typeof(*), ::Type{α}, ::Type{A}, ::Type{B}) where {α<:Number, A<:AbstractVector,B<:AbstractMatrix} =
    MulAddStyle()
ApplyStyle(::typeof(+), ::Type{<:Applied{MulAddStyle,typeof(*)}}, ::Type{<:Applied{<:Any,typeof(*)}}) = MulAddStyle() # TODO: simpler second arg
ApplyStyle(::typeof(+), ::Type{<:Applied{MulAddStyle,typeof(*)}}, ::Type{<:AbstractArray}) = MulAddStyle()
ApplyStyle(::typeof(+), ::Type{<:Applied{MulStyle,typeof(*)}}, ::Type{<:Applied{<:Any,typeof(*)}}) = MulAddStyle() # TODO: simpler second arg
ApplyStyle(::typeof(+), ::Type{<:Applied{MulStyle,typeof(*)}}, ::Type{<:AbstractArray}) = MulAddStyle()


_αAB(M::Applied{<:Any,typeof(*),<:Tuple{<:AbstractArray,<:AbstractArray}}, ::Type{T}) where T = tuple(scalarone(T), M.args...)
_αAB(M::Applied{<:Any,typeof(*),<:Tuple{<:Number,<:AbstractArray,<:AbstractArray}}, ::Type{T}) where T = M.args
_αABβC(M::Applied{<:Any,typeof(*)}, ::Type{T}) where T = tuple(_αAB(M, T)..., scalarzero(T), fillzeros(T,axes(M)))

_βC(M::Applied{<:Any,typeof(*)}, ::Type{T}) where T = M.args
_βC(M::AbstractArray, ::Type{T}) where T = (scalarone(T), M)

_αABβC(M::Applied{MulAddStyle,typeof(+)}, ::Type{T}) where T =
    tuple(_αAB(M.args[1], T)..., _βC(M.args[2], T)...)

MulAdd(M::Applied) = MulAdd(_αABβC(M, eltype(M))...)

similar(M::Applied{MulAddStyle}, ::Type{T}) where T = similar(MulAdd(M), T)
copy(M::Applied{MulAddStyle}) = copy(MulAdd(M))
@inline copyto!(dest::AbstractArray, M::Applied{MulAddStyle}) = copyto!(dest, MulAdd(M))

