module LazyLinearAlgebra
using Base
using LinearAlgebra

import Base: ReinterpretArray, ReshapedArray, AbstractCartesianIndex, Slice,
             RangeIndex

export Mul

include("memorylayout.jl")

struct Mul{StyleA<:MemoryLayout, StyleX<:MemoryLayout, AType, XType}
    style_A::StyleA
    style_x::StyleX
    A::AType
    x::XType
end

Mul(A, x) = Mul(MemoryLayout(A), MemoryLayout(x), A, x)


end # module
