module LazyLinearAlgebra
using Base
using Base.Broadcast
using LinearAlgebra
using LinearAlgebra.BLAS

import Base: ReinterpretArray, ReshapedArray, AbstractCartesianIndex, Slice,
             RangeIndex, BroadcastStyle, copyto!, length, broadcastable, axes,
             getindex, eltype, tail
import Base.Broadcast: DefaultArrayStyle, Broadcasted, instantiate
import LinearAlgebra.BLAS: BlasFloat, BlasReal, BlasComplex
export Mul

include("memorylayout.jl")
include("mul.jl")
# include("ldiv.jl")

end # module
