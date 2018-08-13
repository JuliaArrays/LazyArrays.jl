# LazyArrays.jl

[![Travis](https://travis-ci.org/JuliaArrays/LazyArrays.jl.svg?branch=master)](https://travis-ci.org/JuliaArrays/LazyArrays.jl)
[![codecov](https://codecov.io/gh/JuliaArrays/LazyArrays.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaArrays/LazyArrays.jl)

Lazy arrays and linear algebra in Julia v0.7

This package supports lazy analogues of array operations like `vcat`, `hcat`,
and multiplication. This helps with the implementation of matrix-free methods
for iterative solvers.

The package has been designed with high-performance in mind, so should outperform
the non-lazy analogues from Base for many operations like `copyto!` and broadcasting.
Some operations will be inherently slower due to extra computation, like `getindex`.
Please file an issue for any examples that are significantly slower than their
the analogue in Base.

## Concatenation

`Vcat` is the lazy analogue of `vcat`. For lazy vectors like ranges, it
creating such a vector is allocation-free. `copyto!` allows for allocation-free
population of a vector.
```julia
julia> using LazyArrays, BenchmarkTools

julia> A = Vcat(1:5,2:3) # allocation-free
7-element Vcat{Int64,1,Tuple{UnitRange{Int64},UnitRange{Int64}}}:
 1
 2
 3
 4
 5
 2
 3

julia> Vector(A) == vcat(1:5, 2:3)
true

julia> b = Array{Int}(undef, length(A)); @btime copyto!(b, A);
  26.670 ns (0 allocations: 0 bytes)

julia> @btime vcat(1:5, 2:3); # takes twice as long due to memory creation
  43.336 ns (1 allocation: 144 bytes)
```
Similarly, `Hcat` is the lazy analogue of `hcat`.
```julia
julia> A = Hcat(1:3, randn(3,10))
3×11 Hcat{Float64,Tuple{UnitRange{Int64},Array{Float64,2}}}:
 1.0   0.350927   0.339103  -1.03526   …   0.786593    0.0416694
 2.0  -1.10206    1.52817    0.223099      0.851804    0.430933
 3.0  -1.26467   -0.743712  -0.828781     -0.0637502  -0.066743

julia> Matrix(A) == hcat(A.arrays...)
true

julia> b = Array{Int}(undef, length(A)); @btime copyto!(b, A);
  26.670 ns (0 allocations: 0 bytes)

julia> B = Array{Float64}(undef, size(A)...); @btime copyto!(B, A);
  109.625 ns (1 allocation: 32 bytes)

julia> @btime hcat(A.arrays...); # takes twice as long due to memory creation
  274.620 ns (6 allocations: 560 bytes)
```

## Broadcasting

Base now includes a lazy broadcast object called `Broadcasting`, but this is
not a subtype of `AbstractArray`. Here we have `BroadcastArray` which replicates
the functionality of `Broadcasting` while supporting the array interface.
```julia
julia> A = randn(6,6);

julia> B = BroadcastArray(exp, A);

julia> Matrix(B) == exp.(A)
true

julia> B = BroadcastArray(+, A, 2);

julia> B == A .+ 2
true
```

## Multiplication

Following Base's lazy broadcasting, we introduce lazy multiplication. The type
`Mul` support multiplication of any two objects, not necessarily arrays.
(In the future there will be a `MulArray` a la `BroadcastArray`.)

`Mul` is designed to work along with broadcasting, and to lower to BLAS calls
whenever possible:
```julia
julia> A = randn(5,5); b = randn(5); c = randn(5); d = similar(c);

julia> d .= 2.0 .* Mul(A,b) .+ 3.0 .* c # Calls gemv!
5-element Array{Float64,1}:
 -2.5366335879717514
 -5.305097174484744  
 -9.818431932350942  
  2.421562605495651  
  0.26792916096572983

julia> 2*(A*b) + 3c
5-element Array{Float64,1}:
 -2.5366335879717514
 -5.305097174484744  
 -9.818431932350942  
  2.421562605495651  
  0.26792916096572983

julia> function mymul(A, b, c, d) # need to put in function for benchmarking
       d .= 2.0 .* Mul(A,b) .+ 3.0 .* c
       end
mymul (generic function with 1 method)

julia> @btime mymul(A, b, c, d) # calls gemv!
  77.444 ns (0 allocations: 0 bytes)
5-element Array{Float64,1}:
 -2.5366335879717514
 -5.305097174484744  
 -9.818431932350942  
  2.421562605495651  
  0.26792916096572983

julia> @btime 2*(A*b) + 3c; # does not call gemv!
  241.659 ns (4 allocations: 512 bytes)
```
