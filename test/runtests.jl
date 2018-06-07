using Test, LinearAlgebra, LazyLinearAlgebra

A = randn(5,5); b = randn(5); c = similar(b);

c .= Mul(A,b)
@test all(c .=== BLAS.gemv!('N', 1.0, A, b, 0.0, similar(c)))

c .= 2.0 .* Mul(A,b)
@test all(c .=== BLAS.gemv!('N', 2.0, A, b, 0.0, similar(c)))

c = copy(b)
c .= Mul(A,b) .+ c
@test all(c .=== BLAS.gemv!('N', 1.0, A, b, 1.0, copy(b)))


c = copy(b)
c .= Mul(A,b) .+ 2.0 .* c
@test all(c .=== BLAS.gemv!('N', 1.0, A, b, 2.0, copy(b)))


c = copy(b)
c .= 3.0 .* Mul(A,b) .+ 2.0 .* c
@test all(c .=== BLAS.gemv!('N', 3.0, A, b, 2.0, copy(b)))

d = similar(c)
c = copy(b)
d .= 3.0 .* Mul(A,b) .+ 2.0 .* c
@test all(d .=== BLAS.gemv!('N', 3.0, A, b, 2.0, copy(b)))
