using Test, LinearAlgebra, LazyArrays, FillArrays, ArrayLayouts, SparseArrays
using StaticArrays
import LazyArrays: CachedArray, colsupport, rowsupport, LazyArrayStyle, broadcasted,
            ApplyLayout, BroadcastLayout, AddArray, LazyLayout, PaddedLayout, PaddedRows, PaddedColumns
import ArrayLayouts: OnesLayout

using Aqua
downstream_test = "--downstream_integration_test" in ARGS
@testset "Project quality" begin
    Aqua.test_all(LazyArrays, ambiguities=false, piracies=false,
        stale_deps=!downstream_test)
end

# this should only be included once to avoid method overwritten warnings, as this commites type-piracy
# we include this at the top-level, so that other sub-modules may reuse the module instead of having to include the file
include("infinitearrays.jl")

@testset "Lazy MemoryLayout" begin
    @testset "ApplyArray" begin
        A = [1.0 2; 3 4]
        @test eltype(AddArray(A, Fill(0, (2, 2)), Zeros(2, 2))) == Float64
        @test @inferred(MemoryLayout(typeof(AddArray(A, Fill(0, (2, 2)), Zeros(2, 2))))) ==
            ApplyLayout{typeof(+)}()
    end

    @testset "BroadcastArray" begin
        A = [1.0 2; 3 4]

        @test @inferred(MemoryLayout(typeof(BroadcastArray(+, A, Fill(0, (2, 2)), Zeros(2, 2))))) ==
            BroadcastLayout{typeof(+)}()

        @test MemoryLayout(typeof(Diagonal(BroadcastArray(exp,randn(5))))) == DiagonalLayout{LazyLayout}()
    end

    @testset "adjtrans/symherm/triangular" begin
        A = ApplyArray(+, randn(2,2), randn(2,2))
        B = ApplyArray(+, randn(2,2), im*randn(2,2))
        @test MemoryLayout(Symmetric(A)) isa SymmetricLayout{LazyLayout}
        @test MemoryLayout(Symmetric(B)) isa SymmetricLayout{LazyLayout}
        @test MemoryLayout(Hermitian(A)) isa SymmetricLayout{LazyLayout}
        @test MemoryLayout(Hermitian(B)) isa HermitianLayout{LazyLayout}
        @test MemoryLayout(A') isa LazyLayout
        @test MemoryLayout(B') isa LazyLayout
        @test MemoryLayout(transpose(A)) isa LazyLayout
        @test MemoryLayout(transpose(B)) isa LazyLayout
        @test MemoryLayout(UpperTriangular(A)) isa TriangularLayout{'U', 'N', LazyLayout}
        @test MemoryLayout(UpperTriangular(A)') isa TriangularLayout{'L', 'N', LazyLayout}
        @test MemoryLayout(UpperTriangular(B)) isa TriangularLayout{'U', 'N', LazyLayout}
        @test MemoryLayout(UpperTriangular(B)') isa TriangularLayout{'L', 'N', LazyLayout}
    end
end

include("applytests.jl")
include("multests.jl")
include("ldivtests.jl")
include("addtests.jl")
include("setoptests.jl")
include("macrotests.jl")
include("lazymultests.jl")
include("concattests.jl")
include("paddedtests.jl")
include("broadcasttests.jl")
include("cachetests.jl")

@testset "Kron"  begin
    A = [1,2,3]
    B = [4,5,6,7]

    @test Array(@inferred(Kron(A))) == A == copyto!(similar(A), Kron(A))
    K = @inferred(Kron(A,B))
    @test size(K) == (12,)
    @test size(K,1) == 12
    @test size(K,2) == 1
    @test axes(K) == (Base.OneTo(12),)
    @test [K[k] for k=1:length(K)] == Array(K) == kron(A,B)

    A = randn(3)
    K = @inferred(Kron(A,B))
    @test K isa Kron{Float64}
    @test all(K.args .=== (A,B))
    @test [K[k] for k=1:length(K)] == Array(K) == Array(Kron{Float64}(A,B)) == kron(A,B) == copyto!(similar(K), K)

    C = [7,8,9,10,11]
    K = Kron(A,B,C)
    @test [K[k] for k=1:length(K)] == Array(K) == kron(A,B,C) == copyto!(similar(K), K)

    @testset "Rectangular Factors: Compatibility with Matrix functions" begin
        A = randn(3,2)
        B = randn(4,6)
        K, k = Kron(A,B), kron(A,B)
        @test [K[k,j] for k=1:size(K,1), j=1:size(K,2)] == Array(Kron(A,B)) == k
        @test det(K) == 0  # kronecker of rectangular factors
        @test logdet(K) == -Inf
        @test logabsdet(K) == (-Inf, 0)
        @test isapprox(det(k), det(K); atol=eps(eltype(K)), rtol=0)
        @test tr(K) ≈ tr(k)

        @test K' == Kron(A', B') == k'
        @test transpose(K) == Kron(transpose(A), transpose(B)) == transpose(k)
        @test pinv(K) == Kron(pinv(A), pinv(B)) ≈ pinv(k)
        @test_throws SingularException inv(K)

        K, k = Kron(A,B'), kron(A,B')
        @test [K[k,j] for k=1:size(K,1), j=1:size(K,2)] == Array(Kron(A,B')) == k
        @test_throws DimensionMismatch det(K)
        @test_throws DimensionMismatch tr(K)

        K, k = Kron(A',B), kron(A',B)
        @test [K[k,j] for k=1:size(K,1), j=1:size(K,2)] == Array(Kron(A',B)) == k
        @test_throws DimensionMismatch det(K)
        @test_throws DimensionMismatch tr(K)

        K, k = Kron(A',B'), kron(A',B')
        @test [K[k,j] for k=1:size(K,1), j=1:size(K,2)] == Array(Kron(A',B')) == k
        @test det(K) == 0  # kronecker of rectangular factors
        @test logdet(K) == -Inf
        @test logabsdet(K) == (-Inf, 0)
        @test isapprox(det(k), det(K); atol=eps(eltype(K)), rtol=0)
        @test tr(K) ≈ tr(k)
    end

    @testset "Square Factors: Compatibility with Matrix functions" begin
        # need A,B,C to be invertible
        A, B, C = 0, 0, 0
        while det(A) == 0
            A = randn(3,3)
        end
        while det(B) == 0
            B = randn(6,6)
        end
        while det(C) == 0
            C = randn(2,2)
        end
        K, k = Kron(A,B,C), kron(A,B,C)
        @test [K[k,j] for k=1:size(K,1), j=1:size(K,2)] == Array(Kron(A,B,C)) == k

        @test K' == Kron(A', B', C') == k'
        @test transpose(K) == Kron(transpose(A), transpose(B), transpose(C)) == transpose(k)
        @test pinv(K) == Kron(pinv(A), pinv(B), pinv(C)) ≈ pinv(k)
        @test inv(K) == Kron(inv(A), inv(B), inv(C)) ≈ inv(k)

        @test logdet(K) ≈ logdet(k)
        @test all(logabsdet(K) .≈ logabsdet(k))
        @test det(K) ≈ det(k)
        @test diag(K) ≈ diag(k)
        @test tr(K) ≈ tr(k)
    end

    @testset "Zero Factor" begin
        A = zeros(3,3)
        B = randn(4,4)
        K, k = Kron(A, B), kron(A, B)
        @test [K[k,j] for k=1:size(K,1), j=1:size(K,2)] == Array(Kron(A,B)) == k == zeros(12, 12)
        @test det(K) ≈ det(k) ≈ 0
        @test isinf(logdet(K))
        @test logdet(K) ≈ logdet(k) == -Inf
        @test all(logabsdet(K) .≈ logabsdet(k) .≈ (-Inf, 0))

        c = randn(10)
        @test_throws DimensionMismatch K * c
        C = randn(10, 12)
        @test_throws DimensionMismatch K * C
        c = randn(12)
        @test (K * c) == zeros(12)
    end

    @testset "Identity Factor" begin
        A = I(3)
        B = randn(4,4)
        K, k = Kron(A, B), kron(A, B)
        @test [K[k,j] for k=1:size(K,1), j=1:size(K,2)] == Array(Kron(A,B)) == k
        @test det(K) ≈ det(k) ≈ det(B)^size(A,1)

        c = randn(12)
        @test (K * c) ≈ (k * c)
    end

    A = randn(3,2)
    B = randn(4,6)
    K = Kron(A,B)
    C = similar(K)
    copyto!(C, K)
    @test C == kron(A,B)
    @test_throws DimensionMismatch randn(2,2) .= K

    A = rand(Int,3,2)
    K = Kron(A,B)
    @test K isa Kron{Float64}
    @test all(K.args .=== (A,B))
    @test [K[k,j] for k=1:size(K,1), j=1:size(K,2)] == Array(K) == Array(Kron{Float64}(A,B)) == kron(A,B)

    K = @inferred(Kron{Float64}(Eye{Float64}(1), zeros(4)))
    @test Array(K) == zeros(4,1)

    @testset "Applied bug" begin
        A = randn(5,5)
        K = Kron(A,A)
        k = applied(kron,A,A)
        @test K[1,1] == k[1,1] == A[1,1]^2
        x = randn(5)
        K = Kron(A,x)
        k = applied(kron,A,x)
        @test K[1,1] == k[1,1] == A[1,1]*x[1]
        K = Kron(x,x)
        k = applied(kron,x,x)
        @test K[1] == k[1] == x[1]^2
        K = Kron(A)
        k = applied(kron,A)
        @test K[1,1] == k[1,1] == A[1,1]
        K = Kron(x)
        k = applied(kron,x)
        @test K[1] == k[1] == x[1]
    end

    @testset "triple vector" begin
        x = randn(5)
        y = randn(6)
        z = randn(4)
        K = Kron(x,y,z)
        @test K[1] == K[1,1] == x[1]y[1]z[1]
        @test K == kron(x,y,z)
    end

    @testset "2-factor kron-mul" begin
        A, B = randn(4, 4), randn(3, 2)
        K, k = Kron(A, B), kron(A, B)
        x = randn(size(k, 2))
        X = randn(size(k, 2), 7)

        res_vec = K * x
        @test size(res_vec, 1) == size(K, 1) == size(k, 1)
        @test res_vec ≈ (k * x)

        res_mat = K * X
        @test size(res_mat, 1) == size(K, 1) == size(k, 1)
        @test res_mat ≈ (k * X)
    end

    @testset "3-factor kron-mul" begin
        A, B, C = randn(4, 4), randn(3, 2), randn(5, 6)
        K, k = Kron(A, B, C), kron(A, B, C)
        x = randn(size(k, 2))
        X = randn(size(k, 2), 8)

        res_vec = K * x
        @test size(res_vec, 1) == size(K, 1) == size(k, 1)
        @test res_vec ≈ (k * x)

        res_mat = K * X
        @test size(res_mat, 1) == size(K, 1) == size(k, 1)
        @test res_mat ≈ (k * X)
    end

    @testset "3-factor all-square kron-mul" begin
        A, B, C = randn(4, 4), randn(3, 3), randn(5, 5)
        K, k = Kron(A, B, C), kron(A, B, C)
        x = randn(size(k, 2))
        X = randn(size(k, 2), 8)

        res_vec = K * x
        @test size(res_vec, 1) == size(K, 1) == size(k, 1)
        @test res_vec ≈ (k * x)

        res_mat = K * X
        @test size(res_mat, 1) == size(K, 1) == size(k, 1)
        @test res_mat ≈ (k * X)
    end

    @testset "3-factor sparse kron-mul" begin
        A, B, C = sprandn(4, 4, 0.2), sprandn(3, 3, 0.2), sprandn(5, 5, 0.2)
        K, k = Kron(A, B, C), kron(A, B, C)
        x = randn(size(k, 2))
        X = randn(size(k, 2), 8)

        res_vec = K * x
        @test size(res_vec, 1) == size(K, 1) == size(k, 1)
        @test res_vec ≈ (k * x)

        res_mat = K * X
        @test size(res_mat, 1) == size(K, 1) == size(k, 1)
        @test res_mat ≈ (k * X)
    end

    @testset "3-factor mixed-sparsity kron-mul" begin
        A, B, C = sprandn(4, 4, 0.2), randn(3, 3), sprandn(5, 5, 0.2)
        K, k = Kron(A, B, C), kron(A, B, C)
        x = randn(size(k, 2))
        X = randn(size(k, 2), 8)

        res_vec = K * x
        @test size(res_vec, 1) == size(K, 1) == size(k, 1)
        @test res_vec ≈ (k * x)

        res_mat = K * X
        @test size(res_mat, 1) == size(K, 1) == size(k, 1)
        @test res_mat ≈ (k * X)
    end

    @testset "kron-by-kron mul" begin
        A, B = randn(3, 4), rand(4, 6)
        C, D = randn(4, 5), rand(6, 4)

        K1, K2 = Kron(A, B), Kron(C, D)
        res = K1 * K2
        @test res ≈ (kron(A, B) * kron(C, D))
        @test res isa Kron{eltype(A), 2}
    end


    @testset "kron-by-kron mul (unaligned)" begin
        A, B = randn(3, 8), rand(4, 3)
        C, D = randn(4, 5), rand(6, 4)

        K1, K2 = Kron(A, B), Kron(C, D)
        res = K1 * K2
        @test res ≈ (kron(A, B) * kron(C, D))
        @test res isa Matrix{eltype(A)}
    end

end

@testset "Diff and Cumsum" begin
    x = Vcat([3,4], [1,1,1,1,1], 3)
    y = @inferred(cumsum(x))
    @test y isa Vcat
    @test y == cumsum(Vector(x)) == Cumsum(x)
    @test @inferred(diff(x)) == diff(Vector(x)) == Diff(x)
    @test diff(x) isa Vcat
    @test Cumsum(x) == Cumsum(x)

    @test sum(x) == sum(Vector(x)) == last(y)
    @test cumsum(Vcat(4)) === Vcat(4)
    @test diff(Vcat(4)) === Vcat{Int}()

    A = randn(3,4)
    @test Cumsum(A; dims=1) == cumsum(A; dims=1)
    @test Cumsum(A; dims=2) == cumsum(A; dims=2)
    @test Diff(A; dims=1) == diff(A; dims=1)
    @test Diff(A; dims=2) == diff(A; dims=2)

    @test Cumsum(A; dims=1) == Cumsum(A; dims=1)

    a = Vcat([1,2,3], Fill(2,100_000_000))
    @test @inferred(cumsum(a))[end] == sum(a) == 200000006
    a = Vcat(2, Fill(2,100_000_000))
    @test @inferred(cumsum(a))[end] == sum(a) == 200000002

    @testset "empty" begin
        @test cumsum(Vcat(Int[], 1:5)) == cumsum(1:5)
        a = Vcat(Int[], [1,2,3])
        @test @inferred(cumsum(a)) == cumsum(Vector(a))
        a = Vcat(1, [1,2], [4,5,6])
        @test @inferred(cumsum(a)) == cumsum(Vector(a))
        a = Vcat(1, Int[], [4,5,6])
        @test cumsum(a) == cumsum(Vector(a))
        a = Vcat(1, Int[], Int[], [4,5,6])
        @test cumsum(a) == cumsum(Vector(a))
    end

    @testset "lazy cumsum" begin
        c = BroadcastArray(exp, 1:10)
        @test cumsum(c) == Cumsum(BroadcastArray(exp, 1:10))
        @test cumsum(BroadcastArray(exp, 1:10)) isa typeof(Cumsum(BroadcastArray(exp, 1:10)))
        @test cumsum(ApplyArray(+, 1:10)) == Cumsum(ApplyArray(+, 1:10))
        @test cumsum(ApplyArray(+, 1:10)) isa typeof(Cumsum(ApplyArray(+, 1:10)))

        @test copyto!(similar(c), cumsum(c)) == cumsum(Vector(c))
    end

    @testset "Cumprod" begin
        a = Accumulate(*, 1:5)
        @test IndexStyle(typeof(a)) == IndexLinear()
        @test a == cumprod(1:5)
        v = BroadcastArray(+, 1, BroadcastArray(^, 1:10_000_000, -2.0))
        a = accumulate(*, v)
        @test a isa Accumulate
        @test copy(a) isa Accumulate
        @test copy(a) == a
        @test a[end] ≈ prod(1 .+ (1:10_000_000).^(-2.0))
        @test LazyArrays.AccumulateAbstractVector(*, 1:5) == Accumulate(*, 1:5)
        @test LazyArrays.AccumulateAbstractVector(*, 1:5) isa LazyArrays.AccumulateAbstractVector
    end
end

@testset "col/rowsupport" begin
    A = randn(5,6)
    @test rowsupport(A,1) === Base.OneTo(6)
    @test colsupport(A,1) === Base.OneTo(5)
    D = Diagonal(randn(5))
    @test rowsupport(D,3) === colsupport(D,3) === 3
    Z = Zeros(5)
    @test rowsupport(Z,1) === colsupport(Z,1) === 1:0

    C = cache(Array,D);
    @test colsupport(C,2) === 2:2
    @test @inferred(colsupport(C,1)) === 1:1
    @test colsupport(cache(Zeros(5,5)),1) == 1:0
    C = cache(Zeros(5));
    @test colsupport(C,1) == 1:0
    C[3] = 1
    @test colsupport(C,1) == 1:3

    LazyArrays.zero!(C)
    @test colsupport(C,1) == 1:3
    @test C == zeros(5)

    @testset "convexunion" begin
        # bug from BandedMartrices.jl
        @test LazyArrays.convexunion(7:10,9:8) == LazyArrays.convexunion(9:8,7:10) == 7:10
        @test LazyArrays.convexunion(1,5) == LazyArrays.convexunion(1:3,5) == LazyArrays.convexunion(5,1:3) == 1:5
    end
end

@testset "triu/tril" begin
    A = ApplyArray(triu,randn(2,2))
    @test A isa ApplyArray{Float64}
    @test A[2,1] == 0
    @test A[1,1] == A.args[1][1,1]
    @test A == triu(A.args[1])
    @test size(A) == (2,2)
    A = ApplyArray(tril,randn(2,2))
    @test A isa ApplyArray{Float64}
    @test A[1,2] == 0
    @test A[1,1] == A.args[1][1,1]
    @test A == tril(A.args[1])
    A = ApplyArray(triu,randn(2,2),1)
    @test A[1,1] == 0
    @test A[1,2] == A.args[1][1,2]
    @test A isa ApplyArray{Float64}
    @test A == triu(A.args[1],1)
    A = ApplyArray(tril,randn(2,2),-1)
    @test A isa ApplyArray{Float64}
    @test A[1,1] == 0
    @test A[2,1] == A.args[1][2,1]
    @test A == tril(A.args[1],-1)

    A = ApplyMatrix(exp,randn(2,2))
    @test triu(A) isa ApplyMatrix{Float64,typeof(triu)}
    @test tril(A) isa ApplyMatrix{Float64,typeof(tril)}
    @test triu(A,1) isa ApplyMatrix{Float64,typeof(triu)}
    @test tril(A,1) isa ApplyMatrix{Float64,typeof(tril)}
end

@testset "BroadcastArray" begin
    bc = broadcasted(exp,[1,2,3])
    v = BroadcastArray(exp, [1,2,3])
    @test BroadcastArray(bc) == BroadcastVector(bc) == BroadcastVector{Float64,typeof(exp),typeof(bc.args)}(bc) ==
        v == BroadcastVector(exp, [1,2,3]) == exp.([1,2,3])

    Base.IndexStyle(typeof(BroadcastVector(exp, [1,2,3]))) == IndexLinear()

    bc = broadcasted(exp,[1 2; 3 4])
    M = BroadcastArray(exp, [1 2; 3 4])
    @test BroadcastArray(bc) == BroadcastMatrix(bc) == BroadcastMatrix{Float64,typeof(exp),typeof(bc.args)}(bc) ==
        M == BroadcastMatrix(BroadcastMatrix(bc)) == BroadcastMatrix(exp,[1 2; 3 4]) == exp.([1 2; 3 4])

    @test exp.(v') isa Adjoint{<:Any,<:BroadcastVector}
    @test exp.(transpose(v)) isa Transpose{<:Any,<:BroadcastVector}
    @test exp.(M') isa Adjoint{<:Any,<:BroadcastMatrix}
    @test exp.(transpose(M)) isa Transpose{<:Any,<:BroadcastMatrix}

    bc = BroadcastArray(broadcasted(+, 1:10, broadcasted(sin, 1:10)))
    @test bc[1:10] == (1:10) .+ sin.(1:10)

    bc = BroadcastArray(broadcasted(+,1:10,broadcasted(+,1,2)))
    @test bc.args[2] == 3
end

@testset "_vec_mul_arguments method" begin
    @test_throws "MethodError: no method matching _vec_mul_arguments"  LazyArrays._vec_mul_arguments(2, [])
end

include("blocktests.jl")
include("bandedtests.jl")
include("blockbandedtests.jl")
