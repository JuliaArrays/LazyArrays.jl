using Test, LinearAlgebra, LazyArrays, StaticArrays, FillArrays, ArrayLayouts
import LazyArrays: CachedArray, colsupport, rowsupport, LazyArrayStyle, broadcasted,
            PaddedLayout, ApplyLayout, BroadcastLayout, AddArray, LazyLayout

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

    @testset "Vcat" begin
        @test @inferred(MemoryLayout(typeof(Vcat(Ones(10),Zeros(10))))) == PaddedLayout{FillLayout}()
        @test @inferred(MemoryLayout(typeof(Vcat([1.],Zeros(10))))) == PaddedLayout{DenseColumnMajor}()
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

    A = randn(3,2)
    B = randn(4,6)
    K, k = Kron(A,B), kron(A,B)
    @test [K[k,j] for k=1:size(K,1), j=1:size(K,2)] == Array(Kron(A,B)) == k
    @test det(K) == 0  # kronecker of rectangular factors
    @test isapprox(det(k), det(K); atol=eps(eltype(K)), rtol=0)
    @test tr(K) ≈ tr(k)

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
    @test isapprox(det(k), det(K); atol=eps(eltype(K)), rtol=0)
    @test tr(K) ≈ tr(k)

    A = randn(3,3)
    B = randn(6,6)
    C = randn(2,2)
    K, k = Kron(A,B,C), kron(A,B,C)
    @test [K[k,j] for k=1:size(K,1), j=1:size(K,2)] == Array(Kron(A,B,C)) == k
    @test det(K) ≈ det(k)
    @test tr(K) ≈ tr(k)

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

    @test_broken cumsum(Vcat(Int[], 1:5)) == cumsum(1:5)

    @test cumsum(BroadcastArray(exp, 1:10)) === Cumsum(BroadcastArray(exp, 1:10))
    @test cumsum(ApplyArray(+, 1:10)) === Cumsum(ApplyArray(+, 1:10))
end

@testset "col/rowsupport" begin
    A = randn(5,6)
    @test rowsupport(A,1) === Base.OneTo(6)
    @test colsupport(A,1) === Base.OneTo(5)
    D = Diagonal(randn(5))
    @test rowsupport(D,3) === colsupport(D,3) === 3:3
    Z = Zeros(5)
    @test rowsupport(Z,1) === colsupport(Z,1) === 1:0
    @test_broken cache(D)
    C = cache(Array,D);
    @test colsupport(C,2) === 2:2
    @test colsupport(C,1) === 1:1
    @test colsupport(cache(Zeros(5,5)),1) == 1:0
    C = cache(Zeros(5));
    @test colsupport(C,1) == 1:0
    C[3] = 1
    @test colsupport(C,1) == 1:3

    LazyArrays.zero!(C)
    @test colsupport(C,1) == 1:3
    @test C == zeros(5)

    # bug from BandedMartrices.jl
    @test LazyArrays.convexunion(7:10,9:8) == LazyArrays.convexunion(9:8,7:10) == 7:10
end

@testset "triu/tril" begin
    A = ApplyArray(triu,randn(2,2))
    @test A isa ApplyArray{Float64}
    @test A[2,1] == 0
    @test A[1,1] == A.args[1][1,1]
    @test A == triu(A.args[1])
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

    @test exp.(v') isa BroadcastMatrix
    @test exp.(transpose(v)) isa BroadcastMatrix
    @test exp.(M') isa BroadcastMatrix
    @test exp.(transpose(M)) isa BroadcastMatrix

    bc = BroadcastArray(broadcasted(+, 1:10, broadcasted(sin, 1:10)))
    @test bc[1:10] == (1:10) .+ sin.(1:10)
    
    bc = BroadcastArray(broadcasted(+,1:10,broadcasted(+,1,2)))
    @test bc.args[2] == 3
end