var documenterSearchIndex = {"docs":
[{"location":"#LazyArrays.jl-1","page":"Home","title":"LazyArrays.jl","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"Modules = [LazyArrays]\nPrivate = false","category":"page"},{"location":"#LazyArrays.LazyArrays","page":"Home","title":"LazyArrays.LazyArrays","text":"LazyArrays.jl\n\n(Image: Dev) (Image: Travis) (Image: codecov)\n\nLazy arrays and linear algebra in Julia\n\nThis package supports lazy analogues of array operations like vcat, hcat, and multiplication. This helps with the implementation of matrix-free methods for iterative solvers.\n\nThe package has been designed with high-performance in mind, so should outperform the non-lazy analogues from Base for many operations like copyto! and broadcasting. Some operations will be inherently slower due to extra computation, like getindex. Please file an issue for any examples that are significantly slower than their the analogue in Base.\n\nConcatenation\n\nVcat is the lazy analogue of vcat. For lazy vectors like ranges, it creating such a vector is allocation-free. copyto! allows for allocation-free population of a vector.\n\njulia> using LazyArrays, BenchmarkTools\n\njulia> A = Vcat(1:5,2:3) # allocation-free\n7-element Vcat{Int64,1,Tuple{UnitRange{Int64},UnitRange{Int64}}}:\n 1\n 2\n 3\n 4\n 5\n 2\n 3\n\njulia> Vector(A) == vcat(1:5, 2:3)\ntrue\n\njulia> b = Array{Int}(undef, length(A)); @btime copyto!(b, A);\n  26.670 ns (0 allocations: 0 bytes)\n\njulia> @btime vcat(1:5, 2:3); # takes twice as long due to memory creation\n  43.336 ns (1 allocation: 144 bytes)\n\nSimilarly, Hcat is the lazy analogue of hcat.\n\njulia> A = Hcat(1:3, randn(3,10))\n3×11 Hcat{Float64,Tuple{UnitRange{Int64},Array{Float64,2}}}:\n 1.0   0.350927   0.339103  -1.03526   …   0.786593    0.0416694\n 2.0  -1.10206    1.52817    0.223099      0.851804    0.430933\n 3.0  -1.26467   -0.743712  -0.828781     -0.0637502  -0.066743\n\njulia> Matrix(A) == hcat(A.arrays...)\ntrue\n\njulia> b = Array{Int}(undef, length(A)); @btime copyto!(b, A);\n  26.670 ns (0 allocations: 0 bytes)\n\njulia> B = Array{Float64}(undef, size(A)...); @btime copyto!(B, A);\n  109.625 ns (1 allocation: 32 bytes)\n\njulia> @btime hcat(A.arrays...); # takes twice as long due to memory creation\n  274.620 ns (6 allocations: 560 bytes)\n\nBroadcasting\n\nBase now includes a lazy broadcast object called Broadcasting, but this is not a subtype of AbstractArray. Here we have BroadcastArray which replicates the functionality of Broadcasting while supporting the array interface.\n\njulia> A = randn(6,6);\n\njulia> B = BroadcastArray(exp, A);\n\njulia> Matrix(B) == exp.(A)\ntrue\n\njulia> B = BroadcastArray(+, A, 2);\n\njulia> B == A .+ 2\ntrue\n\nSuch arrays can also be created using the macro @~ which acts on ordinary  broadcasting expressions combined with LazyArray:\n\njulia> C = rand(1000)';\n\njulia> D = LazyArray(@~ exp.(C))\n\njulia> E = LazyArray(@~ @. 2 + log(C))\n\njulia> @btime sum(LazyArray(@~ C .* C'); dims=1) # without `@~`, 1.438 ms (5 allocations: 7.64 MiB)\n  74.425 μs (7 allocations: 8.08 KiB)\n\nMultiplication\n\nFollowing Base's lazy broadcasting, we introduce lazy multiplication. The type Mul support multiplication of any two objects, not necessarily arrays. (In the future there will be a MulArray a la BroadcastArray.)\n\nMul is designed to work along with broadcasting, and to lower to BLAS calls whenever possible:\n\njulia> A = randn(5,5); b = randn(5); c = randn(5); d = similar(c);\n\njulia> d .= 2.0 .* Mul(A,b) .+ 3.0 .* c # Calls gemv!\n5-element Array{Float64,1}:\n -2.5366335879717514\n -5.305097174484744  \n -9.818431932350942  \n  2.421562605495651  \n  0.26792916096572983\n\njulia> 2*(A*b) + 3c\n5-element Array{Float64,1}:\n -2.5366335879717514\n -5.305097174484744  \n -9.818431932350942  \n  2.421562605495651  \n  0.26792916096572983\n\njulia> function mymul(A, b, c, d) # need to put in function for benchmarking\n       d .= 2.0 .* Mul(A,b) .+ 3.0 .* c\n       end\nmymul (generic function with 1 method)\n\njulia> @btime mymul(A, b, c, d) # calls gemv!\n  77.444 ns (0 allocations: 0 bytes)\n5-element Array{Float64,1}:\n -2.5366335879717514\n -5.305097174484744  \n -9.818431932350942  \n  2.421562605495651  \n  0.26792916096572983\n\njulia> @btime 2*(A*b) + 3c; # does not call gemv!\n  241.659 ns (4 allocations: 512 bytes)\n\nUsing @~ macro, above expression using Mul can also be written as\n\nd .= @~ 2.0 .* (A * b) .+ 3.0 .* c\n\nInverses\n\nWe also have lazy inverses PInv(A), designed to work alongside Mul to  to lower to BLAS calls whenever possible:\n\njulia> A = randn(5,5); b = randn(5); c = similar(b);\n\njulia> c .= Mul(PInv(A), b)\n5-element Array{Float64,1}:\n -2.5366335879717514\n -5.305097174484744  \n -9.818431932350942  \n  2.421562605495651  \n  0.26792916096572983\n\njulia> c .= Ldiv(A, b) # shorthand for above\n5-element Array{Float64,1}:\n -2.5366335879717514\n -5.305097174484744  \n -9.818431932350942  \n  2.421562605495651  \n  0.26792916096572983\n\nKronecker products\n\nWe can represent Kronecker products of arrays without constructing the full array.\n\njulia> A = randn(2,2); B = randn(3,3);\n\njulia> K = Kron(A,B)\n6×6 Kron{Float64,2,Tuple{Array{Float64,2},Array{Float64,2}}}:\n  1.99255  -1.45132    0.864789  -0.785538   0.572163  -0.340932\n -2.7016    0.360785  -1.78671    1.06507   -0.142235   0.70439\n  1.89938  -2.69996    0.200992  -0.748806   1.06443   -0.0792386\n -1.84225   1.34184   -0.799557  -2.45355    1.7871    -1.06487\n  2.49782  -0.333571   1.65194    3.32665   -0.444258   2.20009\n -1.75611   2.4963    -0.185831  -2.33883    3.32464   -0.247494\n\njulia> C = Matrix{Float64}(undef, 6, 6); @btime copyto!(C, K);\n  61.528 ns (0 allocations: 0 bytes)\n\njulia> C == kron(A,B)\ntrue\n\n\n\n\n\n","category":"module"},{"location":"#LazyArrays.LazyArray","page":"Home","title":"LazyArrays.LazyArray","text":"LazyArray(x::Applied) :: ApplyArray\nLazyArray(x::Broadcasted) :: BroadcastArray\n\nWrap a lazy object that wraps a computation producing an array to an array.\n\n\n\n\n\n","category":"type"},{"location":"#LazyArrays.cache-Union{Tuple{MT}, Tuple{Type{MT},AbstractArray}} where MT<:AbstractArray","page":"Home","title":"LazyArrays.cache","text":"cache(array::AbstractArray)\n\nCaches the entries of an array.\n\n\n\n\n\n","category":"method"},{"location":"#LazyArrays.@~-Tuple{Any}","page":"Home","title":"LazyArrays.@~","text":"@~ expr\n\nMacro for creating a Broadcasted or Applied object.  Regular calls like f(args...) inside expr are replaced with applied(f, args...). Dotted-calls like f(args...) inside expr are replaced with broadcasted.(f, args...).  Use LazyArray(@~ expr) if you need an array-based interface.\n\njulia> @~ A .+ B ./ 2\n\njulia> @~ @. A + B / 2\n\njulia> @~ A * B + C\n\n\n\n\n\n","category":"macro"},{"location":"internals/#Internals-1","page":"Internals","title":"Internals","text":"","category":"section"},{"location":"internals/#","page":"Internals","title":"Internals","text":"Modules = [LazyArrays]\nPublic = false","category":"page"},{"location":"internals/#LazyArrays.AbstractStridedLayout","page":"Internals","title":"LazyArrays.AbstractStridedLayout","text":"AbstractStridedLayout\n\nis an abstract type whose subtypes are returned by MemoryLayout(A) if an array A has storage laid out at regular offsets in memory, and which can therefore be passed to external C and Fortran functions expecting this memory layout.\n\nJulia's internal linear algebra machinery will automatically (and invisibly) dispatch to BLAS and LAPACK routines if the memory layout is BLAS compatible and the element type is a Float32, Float64, ComplexF32, or ComplexF64. In this case, one must implement the strided array interface, which requires overrides of strides(A::MyMatrix) and unknown_convert(::Type{Ptr{T}}, A::MyMatrix).\n\n\n\n\n\n","category":"type"},{"location":"internals/#LazyArrays.Add-Tuple","page":"Internals","title":"LazyArrays.Add","text":"Add(A1, A2, …, AN)\n\nA lazy representation of A1 + A2 + … + AN; i.e., a shorthand for applied(+, A1, A2, …, AN).\n\n\n\n\n\n","category":"method"},{"location":"internals/#LazyArrays.BroadcastLayout","page":"Internals","title":"LazyArrays.BroadcastLayout","text":"BroadcastLayout(f, layouts)\n\nis returned by MemoryLayout(A) if a matrix A is a BroadcastArray. f is a function that broadcast operation is applied and layouts is a tuple of MemoryLayout of the broadcasted arguments.\n\n\n\n\n\n","category":"type"},{"location":"internals/#LazyArrays.ColumnMajor","page":"Internals","title":"LazyArrays.ColumnMajor","text":"ColumnMajor()\n\nis returned by MemoryLayout(A) if an array A has storage in memory as a column major array, so that stride(A,1) == 1 and stride(A,i) ≥ size(A,i-1) * stride(A,i-1) for 2 ≤ i ≤ ndims(A).\n\nArrays with ColumnMajor memory layout must conform to the DenseArray interface.\n\n\n\n\n\n","category":"type"},{"location":"internals/#LazyArrays.DecreasingStrides","page":"Internals","title":"LazyArrays.DecreasingStrides","text":"DecreasingStrides()\n\nis returned by MemoryLayout(A) if an array A has storage in memory as a strided array with decreasing strides, so that stride(A,ndims(A)) ≥ 1 and stride(A,i) ≥ size(A,i+1) * stride(A,i+1)for1 ≤ i ≤ ndims(A)-1`.\n\n\n\n\n\n","category":"type"},{"location":"internals/#LazyArrays.DenseColumnMajor","page":"Internals","title":"LazyArrays.DenseColumnMajor","text":"DenseColumnMajor()\n\nis returned by MemoryLayout(A) if an array A has storage in memory equivalent to an Array, so that stride(A,1) == 1 and stride(A,i) ≡ size(A,i-1) * stride(A,i-1) for 2 ≤ i ≤ ndims(A). In particular, if A is a matrix then strides(A) ==(1, size(A,1))`.\n\nArrays with DenseColumnMajor memory layout must conform to the DenseArray interface.\n\n\n\n\n\n","category":"type"},{"location":"internals/#LazyArrays.DenseRowMajor","page":"Internals","title":"LazyArrays.DenseRowMajor","text":"DenseRowMajor()\n\nis returned by MemoryLayout(A) if an array A has storage in memory as a row major array with dense entries, so that stride(A,ndims(A)) == 1 and stride(A,i) ≡ size(A,i+1) * stride(A,i+1) for 1 ≤ i ≤ ndims(A)-1. In particular, if A is a matrix then strides(A) ==(size(A,2), 1)`.\n\n\n\n\n\n","category":"type"},{"location":"internals/#LazyArrays.HermitianLayout","page":"Internals","title":"LazyArrays.HermitianLayout","text":"HermitianLayout(layout, uplo)\n\nis returned by MemoryLayout(A) if a matrix A has storage in memory as a hermitianized version of layout, where the entries used are dictated by the uplo, which can be 'U' or L'.\n\nA matrix that has memory layout HermitianLayout(layout, uplo) must overrided hermitiandata(A) to return a matrix B such that MemoryLayout(B) == layout and A[k,j] == B[k,j] for j ≥ k if uplo == 'U' (j ≤ k if uplo == 'L') and A[k,j] == conj(B[j,k]) for j < k if uplo == 'U' (j > k if uplo == 'L').\n\n\n\n\n\n","category":"type"},{"location":"internals/#LazyArrays.IncreasingStrides","page":"Internals","title":"LazyArrays.IncreasingStrides","text":"IncreasingStrides()\n\nis returned by MemoryLayout(A) if an array A has storage in memory as a strided array with  increasing strides, so that stride(A,1) ≥ 1 and stride(A,i) ≥ size(A,i-1) * stride(A,i-1) for 2 ≤ i ≤ ndims(A).\n\n\n\n\n\n","category":"type"},{"location":"internals/#LazyArrays.LowerTriangularLayout","page":"Internals","title":"LazyArrays.LowerTriangularLayout","text":"LowerTriangularLayout(layout)\n\nis returned by MemoryLayout(A) if a matrix A has storage in memory equivalent to a LowerTriangular(B) where B satisfies MemoryLayout(B) == layout.\n\nA matrix that has memory layout LowerTriangularLayout(layout) must overrided triangulardata(A) to return a matrix B such that MemoryLayout(B) == layout and A[k,j] ≡ zero(eltype(A)) for j > k and A[k,j] ≡ B[k,j] for j ≤ k.\n\nMoreover, transpose(A) and adjoint(A) must return a matrix that has memory layout UpperTriangularLayout.\n\n\n\n\n\n","category":"type"},{"location":"internals/#LazyArrays.MemoryLayout-Tuple{Any}","page":"Internals","title":"LazyArrays.MemoryLayout","text":"MemoryLayout(A)\n\nspecifies the layout in memory for an array A. When you define a new AbstractArray type, you can choose to override MemoryLayout to indicate how an array is stored in memory. For example, if your matrix is column major with stride(A,2) == size(A,1), then override as follows:\n\nMemoryLayout(::MyMatrix) = DenseColumnMajor()\n\nThe default is UnknownLayout() to indicate that the layout in memory is unknown.\n\nJulia's internal linear algebra machinery will automatically (and invisibly) dispatch to BLAS and LAPACK routines if the memory layout is compatible.\n\n\n\n\n\n","category":"method"},{"location":"internals/#LazyArrays.RowMajor","page":"Internals","title":"LazyArrays.RowMajor","text":"RowMajor()\n\nis returned by MemoryLayout(A) if an array A has storage in memory as a row major array, so that stride(A,ndims(A)) == 1 and stride(A,i) ≥ size(A,i+1) * stride(A,i+1)for1 ≤ i ≤ ndims(A)-1`.\n\nIf A is a matrix  with RowMajor memory layout, then transpose(A) should return a matrix whose layout is ColumnMajor.\n\n\n\n\n\n","category":"type"},{"location":"internals/#LazyArrays.ScalarLayout","page":"Internals","title":"LazyArrays.ScalarLayout","text":"ScalarLayout()\n\nis returned by MemoryLayout(A) if A is a scalar, which does not live in memory\n\n\n\n\n\n","category":"type"},{"location":"internals/#LazyArrays.StridedLayout","page":"Internals","title":"LazyArrays.StridedLayout","text":"StridedLayout()\n\nis returned by MemoryLayout(A) if an array A has storage laid out at regular offsets in memory. Arrays with StridedLayout must conform to the DenseArray interface.\n\n\n\n\n\n","category":"type"},{"location":"internals/#LazyArrays.SymmetricLayout","page":"Internals","title":"LazyArrays.SymmetricLayout","text":"SymmetricLayout(layout, uplo)\n\nis returned by MemoryLayout(A) if a matrix A has storage in memory as a symmetrized version of layout, where the entries used are dictated by the uplo, which can be 'U' or L'.\n\nA matrix that has memory layout SymmetricLayout(layout, uplo) must overrided symmetricdata(A) to return a matrix B such that MemoryLayout(B) == layout and A[k,j] == B[k,j] for j ≥ k if uplo == 'U' (j ≤ k if uplo == 'L') and A[k,j] == B[j,k] for j < k if uplo == 'U' (j > k if uplo == 'L').\n\n\n\n\n\n","category":"type"},{"location":"internals/#LazyArrays.UnitLowerTriangularLayout","page":"Internals","title":"LazyArrays.UnitLowerTriangularLayout","text":"UnitLowerTriangularLayout(ML::MemoryLayout)\n\nis returned by MemoryLayout(A) if a matrix A has storage in memory equivalent to a UnitLowerTriangular(B) where B satisfies MemoryLayout(B) == layout.\n\nA matrix that has memory layout UnitLowerTriangularLayout(layout) must overrided triangulardata(A) to return a matrix B such that MemoryLayout(B) == layout and A[k,j] ≡ zero(eltype(A)) for j > k, A[k,j] ≡ one(eltype(A)) for j == k, A[k,j] ≡ B[k,j] for j < k.\n\nMoreover, transpose(A) and adjoint(A) must return a matrix that has memory layout UnitUpperTriangularLayout.\n\n\n\n\n\n","category":"type"},{"location":"internals/#LazyArrays.UnitUpperTriangularLayout","page":"Internals","title":"LazyArrays.UnitUpperTriangularLayout","text":"UnitUpperTriangularLayout(ML::MemoryLayout)\n\nis returned by MemoryLayout(A) if a matrix A has storage in memory equivalent to a UpperTriangularLayout(B) where B satisfies MemoryLayout(B) == ML.\n\nA matrix that has memory layout UnitUpperTriangularLayout(layout) must overrided triangulardata(A) to return a matrix B such that MemoryLayout(B) == layout and A[k,j] ≡ B[k,j] for j > k, A[k,j] ≡ one(eltype(A)) for j == k, A[k,j] ≡ zero(eltype(A)) for j < k.\n\nMoreover, transpose(A) and adjoint(A) must return a matrix that has memory layout UnitLowerTriangularLayout.\n\n\n\n\n\n","category":"type"},{"location":"internals/#LazyArrays.UnknownLayout","page":"Internals","title":"LazyArrays.UnknownLayout","text":"UnknownLayout()\n\nis returned by MemoryLayout(A) if it is unknown how the entries of an array A are stored in memory.\n\n\n\n\n\n","category":"type"},{"location":"internals/#LazyArrays.UpperTriangularLayout","page":"Internals","title":"LazyArrays.UpperTriangularLayout","text":"UpperTriangularLayout(ML::MemoryLayout)\n\nis returned by MemoryLayout(A) if a matrix A has storage in memory equivalent to a UpperTriangularLayout(B) where B satisfies MemoryLayout(B) == ML.\n\nA matrix that has memory layout UpperTriangularLayout(layout) must overrided triangulardata(A) to return a matrix B such that MemoryLayout(B) == layout and A[k,j] ≡ B[k,j] for j ≥ k and A[k,j] ≡ zero(eltype(A)) for j < k.\n\nMoreover, transpose(A) and adjoint(A) must return a matrix that has memory layout LowerTriangularLayout.\n\n\n\n\n\n","category":"type"},{"location":"internals/#LazyArrays.colsupport-Tuple{Any,Any}","page":"Internals","title":"LazyArrays.colsupport","text":"\"     colsupport(A, j)\n\ngives an iterator containing the possible non-zero entries in the j-th column of A.\n\n\n\n\n\n","category":"method"},{"location":"internals/#LazyArrays.lmaterialize-Tuple{LazyArrays.Applied{Style,typeof(*),Factors} where Factors<:Tuple where Style}","page":"Internals","title":"LazyArrays.lmaterialize","text":"lmaterialize(M::Mul)\n\nmaterializes arrays iteratively, left-to-right.\n\n\n\n\n\n","category":"method"},{"location":"internals/#LazyArrays.rowsupport-Tuple{Any,Any}","page":"Internals","title":"LazyArrays.rowsupport","text":"\"     rowsupport(A, k)\n\ngives an iterator containing the possible non-zero entries in the k-th row of A.\n\n\n\n\n\n","category":"method"}]
}
