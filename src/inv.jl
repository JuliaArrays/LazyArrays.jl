

struct Inv{T, Style, Typ}
    style::Style
    A::Typ
end

Inv(style::S, A::T) where {S,T} = Inv{eltype(A),S,T}(style, A)
Inv(A) = Inv(MemoryLayout(A), A)

eltype(::Inv{T}) where T = T
eltype(::Type{<:Inv{T}}) where T = T

struct InverseLayout{ML} <: MemoryLayout
    layout::ML
end
MemoryLayout(Ai::Inv) = InverseLayout(MemoryLayout(Ai.A))


const Ldiv{T, StyleA, StyleB, AType, BType} =
    Mul{T, InverseLayout{StyleA}, StyleB, Inv{T,StyleA,AType}, BType}

const MixedArrayLdivArray{TV, styleA, styleB, p, q, T, V} =
    Ldiv{TV, styleA, styleB, <:AbstractArray{T,p}, <:AbstractArray{V,q}}

Ldiv(A, B) = Mul(Inv(A), B)

_copyto!(_, dest::AbstractVector, L::MixedArrayLdivArray) = ldiv!(dest, factorize(L.A.A), L.B)
