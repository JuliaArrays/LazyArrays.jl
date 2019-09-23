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

@testset "Kron"  begin
    A = [1,2,3]
    B = [4,5,6,7]

    @test Array(@inferred(Kron(A))) == A
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

@testset "BroadcastArray" begin
    a = randn(6)
    b = BroadcastArray(exp, a)
    @test BroadcastArray(b) == BroadcastVector(b) == b

    @test b ==  Vector(b) == exp.(a)
    @test b[2:5] isa BroadcastVector
    @test b[2:5] == exp.(a[2:5])

    @test exp.(b) isa BroadcastVector
    @test b .+ SVector(1,2,3,4,5,6) isa BroadcastVector
    @test SVector(1,2,3,4,5,6) .+ b isa BroadcastVector

    A = randn(6,6)
    B = BroadcastArray(exp, A)
    
    @test Matrix(B) == exp.(A)


    C = BroadcastArray(+, A, 2)
    @test C == A .+ 2
    D = BroadcastArray(+, A, C)
    @test D == A + C

    @test sum(B) ≈ sum(exp, A)
    @test sum(C) ≈ sum(A .+ 2)
    @test prod(B) ≈ prod(exp, A)
    @test prod(C) ≈ prod(A .+ 2)

    x = Vcat([3,4], [1,1,1,1,1], 1:3)
    @test x .+ (1:10) isa Vcat
    @test (1:10) .+ x isa Vcat
    @test x + (1:10) isa Vcat
    @test (1:10) + x isa Vcat
    @test x .+ (1:10) == (1:10) .+ x == (1:10) + x == x + (1:10) == Vector(x) + (1:10)

    @test exp.(x) isa Vcat
    @test exp.(x) == exp.(Vector(x))
    @test x .+ 2 isa Vcat
    @test (x .+ 2).args[end] ≡ x.args[end] .+ 2 ≡ 3:5
    @test x .* 2 isa Vcat
    @test 2 .+ x isa Vcat
    @test 2 .* x isa Vcat

    A = Vcat([[1 2; 3 4]], [[4 5; 6 7]])
    @test A .+ Ref(I) == Ref(I) .+ A == Vcat([[2 2; 3 5]], [[5 5; 6 8]])

    @test_broken BroadcastArray(*,1.1,[1 2])[1] == 1.1

    B = BroadcastArray(*, Diagonal(randn(5)), randn(5,5))
    @test B == broadcast(*,B.args...)
    @test colsupport(B,1) == rowsupport(B,1) == 1:1
    @test colsupport(B,3) == rowsupport(B,3) == 3:3
    @test colsupport(B,5) == rowsupport(B,5) == 5:5
    B = BroadcastArray(*, Diagonal(randn(5)), 2)
    @test B == broadcast(*,B.args...)
    @test colsupport(B,1) == rowsupport(B,1) == 1:1
    @test colsupport(B,3) == rowsupport(B,3) == 3:3
    @test colsupport(B,5) == rowsupport(B,5) == 5:5
    B = BroadcastArray(*, Diagonal(randn(5)), randn(5))
    @test B == broadcast(*,B.args...)
    @test colsupport(B,1) == rowsupport(B,1) == 1:1
    @test colsupport(B,3) == rowsupport(B,3) == 3:3
    @test colsupport(B,5) == rowsupport(B,5) == 5:5

    B = BroadcastArray(+, Diagonal(randn(5)), 2)
    @test colsupport(B,1) == rowsupport(B,1) == 1:5
    @test colsupport(B,3) == rowsupport(B,3) == 1:5
    @test colsupport(B,5) == rowsupport(B,5) == 1:5
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

@testset "vector*matrix broadcasting #27" begin
    H = [1., 0.]
    @test Mul(H, H') .+ 1 == H*H' .+ 1
    B =  randn(2,2)
    @test Mul(H, H') .+ B == H*H' .+ B
end

@testset "Lazy range" begin
    @test broadcasted(LazyArrayStyle{1}(), +, 1:5) ≡ 1:5
    @test broadcasted(LazyArrayStyle{1}(), +, 1, 1:5) ≡ 2:6
    @test broadcasted(LazyArrayStyle{1}(), +, 1:5, 1) ≡ 2:6

    @test broadcasted(LazyArrayStyle{1}(), +, Fill(2,5)) ≡ Fill(2,5)
    @test broadcasted(LazyArrayStyle{1}(), +, 1, Fill(2,5)) ≡ Fill(3,5)
    @test broadcasted(LazyArrayStyle{1}(), +, Fill(2,5), 1) ≡ Fill(3,5)
    @test broadcasted(LazyArrayStyle{1}(), +, Ref(1), Fill(2,5)) ≡ Fill(3,5)
    @test broadcasted(LazyArrayStyle{1}(), +, Fill(2,5), Ref(1)) ≡ Fill(3,5)
    @test broadcasted(LazyArrayStyle{1}(), +, 1, Fill(2,5)) ≡ Fill(3,5)
    @test broadcasted(LazyArrayStyle{1}(), +, Fill(2,5), Fill(3,5)) ≡ Fill(5,5)

    @test broadcasted(LazyArrayStyle{1}(), *, Zeros(5), Zeros(5)) ≡ Zeros(5)
    b = BroadcastArray(exp, randn(5))
    @test b .* Zeros(5) ≡ Zeros(5)
    @test Zeros(5) .* b ≡ Zeros(5)
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