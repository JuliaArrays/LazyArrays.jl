using Test, LinearAlgebra, LazyArrays, StaticArrays, FillArrays
import LazyArrays: CachedArray, colsupport, rowsupport, LazyArrayStyle, broadcasted

include("memorylayouttests.jl")
include("applytests.jl")
include("multests.jl")
include("ldivtests.jl")
include("addtests.jl")
include("setoptests.jl")
include("macrotests.jl")
include("lazymultests.jl")
include("concattests.jl")
include("broadcasttests.jl")

@testset "Kron"  begin
    A = [1,2,3]
    B = [4,5,6,7]

    @test_throws MethodError Array(@inferred(Kron(A))) == A
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
    @test [K[k] for k=1:length(K)] == Array(K) == Array(Kron{Float64}(A,B)) == kron(A,B)

    # C = [7,8,9,10,11]
    # K = Kron(A,B,C)
    # @time [K[k] for k=1:length(K)] == Array(Kron(A,B)) == kron(A,B)

    A = randn(3,2)
    B = randn(4,6)
    K = Kron(A,B)
    @test [K[k,j] for k=1:size(K,1), j=1:size(K,2)] == Array(Kron(A,B)) == kron(A,B)
    K = Kron(A,B')
    @test [K[k,j] for k=1:size(K,1), j=1:size(K,2)] == Array(Kron(A,B')) == kron(A,B')
    K = Kron(A',B)
    @test [K[k,j] for k=1:size(K,1), j=1:size(K,2)] == Array(Kron(A',B)) == kron(A',B)
    K = Kron(A',B')
    @test [K[k,j] for k=1:size(K,1), j=1:size(K,2)] == Array(Kron(A',B')) == kron(A',B')

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
end

@testset "Cache" begin
    A = 1:10
    C = cache(A)
    @test size(C) == (10,)
    @test axes(C) == (Base.OneTo(10),)
    @test all(Vector(C) .=== Vector(A))
    @test cache(C) isa CachedArray{Int,1,Vector{Int},UnitRange{Int}}
    C2 = cache(C)
    @test C2.data !== C.data
    @test C[:] == C

    @test cache(A)[2,1] == 2
    @test_throws BoundsError cache(A)[2,2]

    A = reshape(1:10^2, 10,10)
    C = cache(A)
    @test size(C) == (10,10)
    @test axes(C) == (Base.OneTo(10),Base.OneTo(10))
    @test all(Array(C) .=== Array(A))
    @test cache(C) isa CachedArray{Int,2,Matrix{Int},typeof(A)}
    C2 = cache(C)
    @test C2.data !== C.data

    A = reshape(1:10^3, 10,10,10)
    C = cache(A)
    @test size(C) == (10,10,10)
    @test axes(C) == (Base.OneTo(10),Base.OneTo(10),Base.OneTo(10))
    @test all(Array(C) .=== Array(A))

    A = reshape(1:10^3, 10,10,10)
    C = cache(A)
    LazyArrays.resizedata!(C,5,5,5)
    LazyArrays.resizedata!(C,8,8,8)
    @test all(C.data .=== Array(A)[1:8,1:8,1:8])

    @test C[1:3,1,:] == A[1:3,1,:]

    @testset "Matrix cache" begin
        A = collect(1:5)
        C = cache(A)
        @test C isa Vector{Int}
        C[1] = 2
        @test A[1] ≠ 2

        A = cache(Matrix(reshape(1:6,2,3)))
        C = cache(A)
        @test C isa Matrix{Int}
        C[1,1] = 2
        @test A[1,1] ≠ 2
        C[1] = 3
        @test A[1,1] ≠ 3
    end

    @testset "setindex!" begin
        A = (1:5)
        C = cache(A)
        C[1] = 2
        @test C[1] == 2

        A = cache(reshape(1:6,2,3))
        C = cache(A)
        C[1,1] = 2
        @test A[1,1] ≠ 2
        @test C[1,1] == 2
        C[1] = 3
        @test C[1,1] == 3

        @test_throws BoundsError C[3,1]
        @test_throws BoundsError C[7]
    end

    @testset "colsupport past size" begin
        C = cache(Zeros(5,5)); C[5,1]; 
        @test colsupport(C,3) == 1:0
    end

    @testset "broadcast" begin
        x = CachedArray([1,2,3],1:8);
        y = 1:8;
        f = (x,y) -> cos(x*y)
        @test f.(x,y) isa CachedArray
        @test @inferred(broadcast(f,x,y)) == f.(Vector(x), Vector(y))

        @test (x + y) isa CachedArray
        @test (x + y).array isa AbstractRange
        @test (x + y) == Vector(x) + Vector(y)

        z = CachedArray([1,4],Zeros{Int}(8));
        @test (x .+ z) isa CachedArray
        @test (x + z) isa CachedArray
        @test Vector( x .+ z) == Vector( x + z) == Vector(x) + Vector(z)

        # Lazy mixed with Static treats as Lazy
        s = SVector(1,2,3,4,5,6,7,8)
        @test f.(x , s) isa CachedArray
        @test f.(x , s) == f.(Vector(x), Vector(s))
    end
end

@testset "Diff and Cumsum" begin
    x = Vcat([3,4], [1,1,1,1,1], 3)
    y = @inferred(cumsum(x))
    @test y isa Vcat
    @test y == cumsum(Vector(x)) == Cumsum(x)
    @test @inferred(diff(x)) == diff(Vector(x)) == Diff(x)
    @test diff(x) isa Vcat

    @test sum(x) == sum(Vector(x)) == last(y)
    @test cumsum(Vcat(4)) === Vcat(4)
    @test diff(Vcat(4)) === Vcat{Int}()

    A = randn(3,4)
    @test Cumsum(A; dims=1) == cumsum(A; dims=1)
    @test Cumsum(A; dims=2) == cumsum(A; dims=2)
    @test Diff(A; dims=1) == diff(A; dims=1)
    @test Diff(A; dims=2) == diff(A; dims=2)

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
    @test A == triu(A.args[1])
    A = ApplyArray(tril,randn(2,2))
    @test A isa ApplyArray{Float64}
    @test A == tril(A.args[1])
    A = ApplyArray(triu,randn(2,2),1)
    @test A isa ApplyArray{Float64}
    @test A == triu(A.args[1],1)
    A = ApplyArray(tril,randn(2,2),-1)
    @test A isa ApplyArray{Float64}
    @test A == tril(A.args[1],-1)
end