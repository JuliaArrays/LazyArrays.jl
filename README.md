# LazyArrays.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliaarrays.github.io/LazyArrays.jl/dev)
[![Build Status](https://github.com/JuliaArrays/LazyArrays.jl/workflows/CI/badge.svg)](https://github.com/JuliaArrays/LazyArrays.jl/actions)
[![codecov](https://codecov.io/gh/JuliaArrays/LazyArrays.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaArrays/LazyArrays.jl)
[![pkgeval](https://juliahub.com/docs/General/LazyArrays/stable/pkgeval.svg)](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/report.html)

Lazy arrays and linear algebra in Julia

This package supports lazy analogues of array operations like `vcat`, `hcat`,
and multiplication. This helps with the implementation of matrix-free methods
for iterative solvers.

The package has been designed with high-performance in mind, so should outperform
the non-lazy analogues from Base for many operations like `copyto!` and broadcasting.
Some operations will be inherently slower due to extra computation, like `getindex`.
Please file an issue for any examples that are significantly slower than their
the analogue in Base.

## Lazy operations

To construct a lazy representation of a function call `f(x,y,z...)`, use the command
`applied(f, x, y, z...)`. This will return an unmaterialized object typically of type `Applied`
that represents the operation. To realize that object, call `materialize`, which 
will typically be equivalent to calling `f(x,y,z...)`. A macro `@~` is available as a shorthand:
```julia
julia> using LazyArrays, LinearAlgebra

julia> applied(exp, 1)
Applied(exp,1)

julia> materialize(applied(exp, 1))
2.718281828459045

julia> materialize(@~ exp(1))
2.718281828459045

julia> exp(1)
2.718281828459045
```

Note that `@~` causes sub-expressions to be wrapped in an `applied` call, not
just the top-level expression. This can lead to `MethodError`s when lazy
application of sub-expressions is not yet implemented. For example,
```julia
julia> @~ Vector(1:10) .* ones(10)'
ERROR: MethodError: ...

julia> A = Vector(1:10); B = ones(10); (@~ sum(A .* B')) |> materialize
550.0
```

The benefit of lazy operations is that they can be materialized in-place, 
possible using simplifications. For example, it is possible to 
do BLAS-like Matrix-Vector operations of the form `α*A*x + β*y` as 
implemented in `BLAS.gemv!` using a lazy applied object:
```julia
julia> A = randn(5,5); b = randn(5); c = randn(5); d = similar(c);

julia> d .= @~ 2.0 * A * b + 3.0 * c # Calls gemv!
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
       d .= @~ 2.0 * A * b + 3.0 * c
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

This also works for inverses, which lower to BLAS calls whenever possible:
```julia
julia> A = randn(5,5); b = randn(5); c = similar(b);

julia> c .= @~ A \ b
5-element Array{Float64,1}:
 -2.5366335879717514
 -5.305097174484744  
 -9.818431932350942  
  2.421562605495651  
  0.26792916096572983
```



## Lazy arrays

Often we want lazy realizations of matrices, which are supported via `ApplyArray`.
For example, the following creates a lazy matrix exponential:
```julia
julia> E = ApplyArray(exp, [1 2; 3 4])
2×2 ApplyArray{Float64,2,typeof(exp),Tuple{Array{Int64,2}}}:
  51.969   74.7366
 112.105  164.074 
```

A lazy matrix exponential is useful for, say, in-place matrix-exponential*vector:
```julia
julia> b = Vector{Float64}(undef, 2); b .= @~ E*[4,4]
2-element Array{Float64,1}:
  506.8220830628333
 1104.7145995988594
```
While this works, it is not actually optimised (yet). 

Other options do have special implementations that make them fast. We now give some examples. 


### Concatenation

Lazy `vcat` and `hcat` allow for representing the concatenation of
vectors without actually allocating memory, and support a fast
`copyto!`  for allocation-free population of a vector.
```julia
julia> using BenchmarkTools

julia> A = ApplyArray(vcat,1:5,2:3) # allocation-free
7-element ApplyArray{Int64,1,typeof(vcat),Tuple{UnitRange{Int64},UnitRange{Int64}}}:
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
Similar is the lazy analogue of `hcat`:
```julia
julia> A = ApplyArray(hcat, 1:3, randn(3,10))
3×11 ApplyArray{Float64,2,typeof(hcat),Tuple{UnitRange{Int64},Array{Float64,2}}}:
 1.0   1.16561    0.224871  -1.36416   -0.30675    0.103714    0.590141   0.982382  -1.50045    0.323747  -1.28173  
 2.0   1.04648    1.35506   -0.147157   0.995657  -0.616321   -0.128672  -0.671445  -0.563587  -0.268389  -1.71004  
 3.0  -0.433093  -0.325207  -1.38496   -0.391113  -0.0568739  -1.55796   -1.00747    0.473686  -1.2113     0.0119156

julia> Matrix(A) == hcat(A.args...)
true

julia> B = Array{Float64}(undef, size(A)...); @btime copyto!(B, A);
  109.625 ns (1 allocation: 32 bytes)

julia> @btime hcat(A.args...); # takes twice as long due to memory creation
  274.620 ns (6 allocations: 560 bytes)
```



### Kronecker products

We can represent Kronecker products of arrays without constructing the full
array:

```julia
julia> A = randn(2,2); B = randn(3,3);

julia> K = ApplyArray(kron,A,B)
6×6 ApplyArray{Float64,2,typeof(kron),Tuple{Array{Float64,2},Array{Float64,2}}}:
 -1.08736   -0.19547   -0.132824   1.60531    0.288579    0.196093 
  0.353898   0.445557  -0.257776  -0.522472  -0.657791    0.380564 
 -0.723707   0.911737  -0.710378   1.06843   -1.34603     1.04876  
  1.40606    0.252761   0.171754  -0.403809  -0.0725908  -0.0493262
 -0.457623  -0.576146   0.333329   0.131426   0.165464   -0.0957293
  0.935821  -1.17896    0.918584  -0.26876    0.338588   -0.26381  

julia> C = Matrix{Float64}(undef, 6, 6); @btime copyto!(C, K);
  61.528 ns (0 allocations: 0 bytes)

julia> C == kron(A,B)
true
```


## Broadcasting

Base includes a lazy broadcast object called `Broadcasting`, but this is
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
Such arrays can also be created using the macro `@~` which acts on ordinary 
broadcasting expressions combined with `LazyArray`:
```julia
julia> C = rand(1000)';

julia> D = LazyArray(@~ exp.(C))

julia> E = LazyArray(@~ @. 2 + log(C))

julia> @btime sum(LazyArray(@~ C .* C'); dims=1) # without `@~`, 1.438 ms (5 allocations: 7.64 MiB)
  74.425 μs (7 allocations: 8.08 KiB)
```

