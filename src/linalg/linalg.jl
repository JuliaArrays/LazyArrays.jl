include("mul.jl")
include("muladd.jl")
include("inv.jl")
include("lazymul.jl")
include("add.jl")


mulapplystyle(::TriangularLayout, ::AbstractStridedLayout) = LmulStyle()
mulapplystyle(::AbstractStridedLayout, ::TriangularLayout) = RmulStyle()