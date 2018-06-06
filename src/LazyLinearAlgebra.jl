module LazyLinearAlgebra
using LinearAlgebra

include("memorylayout.jl")

struct Mul{T, p, StyleA, StyleX, AType, XType} <: AbstractArray{T,p}
    style_A::StyleA
    style_x::StyleX
    A::AType
    x::XType
end



end # module
