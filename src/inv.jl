

struct Inv{T, Style, Typ}
    style::Style
    A::Typ
end

Inv(style::S, A::T) where {S,T} = Inv{eltype(A),S,T}(style, A)
Inv(A) = Inv(MemoryLayout(A), A)

eltype(::Inv{T}) where T = T
eltype(::Type{<:Inv{T}}) where T = T

struct InverseLayout <: MemoryLayout end
MemoryLayout(::Inv) = InverseLayout()


const Ldiv{T, StyleA, StyleB, AType, BType} =
    Mul{T, InverseLayout, StyleB, Inv{T,StyleA,AType}, BType}

Ldiv(A, B) = Mul(Inv(A), B)

copyto!(dest::AbstractVector, L::Ldiv) = ldiv!(dest, factorize(L.A.A), L.B)
