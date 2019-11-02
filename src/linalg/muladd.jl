struct MulAddStyle <: AbstractArrayApplyStyle end
struct ScalarMulStyle <: ApplyStyle end
struct IdentityMulStyle <: AbstractArrayApplyStyle end

combine_mul_styles() = MulAddStyle()
combine_mul_styles(_) = IdentityMulStyle()
combine_mul_styles(::ApplyLayout{typeof(*)}) = FlattenMulStyle()
combine_mul_styles(::ScalarLayout) = ScalarMulStyle()
combine_mul_styles(c1, c2) = result_mul_style(combine_mul_styles(c1), combine_mul_styles(c2))
@inline combine_mul_styles(c1, c2, cs...) = result_mul_style(combine_mul_styles(c1), combine_mul_styles(c2, cs...))

# result_mul_style works on types (singletons and pairs), and leverages `Style`
result_mul_style(_, _) = DefaultArrayApplyStyle()
result_mul_style(::IdentityMulStyle, ::IdentityMulStyle) = MulAddStyle()
result_mul_style(::MulAddStyle, ::MulAddStyle) = DefaultArrayApplyStyle()
result_mul_style(_, ::MulAddStyle) = DefaultArrayApplyStyle()
result_mul_style(::MulAddStyle, _) = DefaultArrayApplyStyle()
result_mul_style(::ScalarMulStyle, S::MulAddStyle) = S
result_mul_style(::MulAddStyle, ::LazyArrayApplyStyle) = LazyArrayApplyStyle()
result_mul_style(::LazyArrayApplyStyle, ::MulAddStyle) = LazyArrayApplyStyle()
result_mul_style(::DefaultArrayApplyStyle, ::LazyArrayApplyStyle) = LazyArrayApplyStyle()
result_mul_style(::LazyArrayApplyStyle, ::DefaultArrayApplyStyle) = LazyArrayApplyStyle()
result_mul_style(::FlattenMulStyle, ::FlattenMulStyle) = FlattenMulStyle()
result_mul_style(::FlattenMulStyle, ::MulAddStyle) = FlattenMulStyle()
result_mul_style(::MulAddStyle, ::FlattenMulStyle) = FlattenMulStyle()
result_mul_style(::FlattenMulStyle, ::LazyArrayApplyStyle) = FlattenMulStyle()
result_mul_style(::LazyArrayApplyStyle, ::FlattenMulStyle) = FlattenMulStyle()
result_mul_style(::FlattenMulStyle, ::DefaultArrayApplyStyle) = FlattenMulStyle()
result_mul_style(::DefaultArrayApplyStyle, ::FlattenMulStyle) = FlattenMulStyle()
result_mul_style(::FlattenMulStyle, ::IdentityMulStyle) = FlattenMulStyle()
result_mul_style(::IdentityMulStyle, ::FlattenMulStyle) = FlattenMulStyle()



@inline mulapplystyle(A...) = combine_mul_styles(A...)

ApplyStyle(::typeof(*), ::Type{α}, ::Type{A}, ::Type{B}) where {α<:Number, A<:AbstractMatrix,B<:AbstractVector} =
    mulapplystyle(MemoryLayout(α), MemoryLayout(A), MemoryLayout(B))
ApplyStyle(::typeof(*), ::Type{α}, ::Type{A}, ::Type{B}) where {α<:Number, A<:AbstractMatrix,B<:AbstractMatrix} =
    mulapplystyle(MemoryLayout(α), MemoryLayout(A), MemoryLayout(B))    
ApplyStyle(::typeof(*), ::Type{α}, ::Type{A}, ::Type{B}) where {α<:Number, A<:AbstractVector,B<:AbstractMatrix} =
    mulapplystyle(MemoryLayout(α), MemoryLayout(A), MemoryLayout(B))        
ApplyStyle(::typeof(+), ::Type{<:Mul{MulAddStyle}}, ::Type{<:Mul}) = MulAddStyle() # TODO: simpler second arg
ApplyStyle(::typeof(+), ::Type{<:Mul{MulAddStyle}}, ::Type{<:AbstractArray}) = MulAddStyle()




scalarone(::Type{T}) where T = one(T)
scalarone(::Type{<:AbstractArray{T}}) where T = scalarone(T)
scalarzero(::Type{T}) where T = zero(T)
scalarzero(::Type{<:AbstractArray{T}}) where T = scalarzero(T)

fillzeros(::Type{T}, ax) where T = Zeros{T}(ax)

_αAB(M::Mul{MulAddStyle,<:Tuple{<:AbstractArray,<:AbstractArray}}, ::Type{T}) where T = tuple(scalarone(T), M.args...)
_αAB(M::Mul{MulAddStyle,<:Tuple{<:Number,<:AbstractArray,<:AbstractArray}}, ::Type{T}) where T = M.args
_αABβC(M::Mul, ::Type{T}) where T = tuple(_αAB(M, T)..., scalarzero(T), fillzeros(T,axes(M)))

_βC(M::Mul, ::Type{T}) where T = M.args
_βC(M::AbstractArray, ::Type{T}) where T = (scalarone(T), M)

_αABβC(M::Applied{MulAddStyle,typeof(+)}, ::Type{T}) where T = 
    tuple(_αAB(M.args[1], T)..., _βC(M.args[2], T)...)

MulAdd(M::Applied) = MulAdd(_αABβC(M, eltype(M))...)

similar(M::Applied{MulAddStyle}, ::Type{T}) where T = similar(MulAdd(M), T)
copy(M::Applied{MulAddStyle}) = copy(MulAdd(M))

@inline copyto!(dest::AbstractArray, M::Applied{MulAddStyle}) = copyto!(dest, MulAdd(M))

###
# DiagonalLayout
###

mulapplystyle(::DiagonalLayout, ::DiagonalLayout) = LmulStyle()

mulapplystyle(::DiagonalLayout, _) = LmulStyle()
mulapplystyle(_, ::DiagonalLayout) = RmulStyle()

diagonallayout(::LazyLayout) = DiagonalLayout{LazyLayout}()
diagonallayout(::ApplyLayout) = DiagonalLayout{LazyLayout}()
diagonallayout(::BroadcastLayout) = DiagonalLayout{LazyLayout}()    
