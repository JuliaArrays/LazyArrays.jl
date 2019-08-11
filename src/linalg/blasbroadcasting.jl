#########
# This file is to add support for lowering broadcast notation
#       y .= α .* Mul(A,x) .+ β .* y
# to
#       materialize!(MulAdd(α, A, x, β, y))
# which then becomes a blas call.
#########


struct MulAddBroadcastStyle <: BroadcastStyle end

# Use default broacasting in general
