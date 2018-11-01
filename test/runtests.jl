using Test, LinearAlgebra, LazyArrays, StaticArrays, FillArrays

include("memorylayouttests.jl")
include("multests.jl")
include("ldivtests.jl")


@testset "concat" begin
    @testset "Vcat" begin
        A = Vcat(Vector(1:10), Vector(1:20))
        @test @inferred(length(A)) == 30
        @test @inferred(A[5]) == A[15] == 5
        @test_throws BoundsError A[31]
        @test reverse(A) == Vcat(Vector(reverse(1:20)), Vector(reverse(1:10)))
        b = Array{Int}(undef, 31)
        @test_throws DimensionMismatch copyto!(b, A)
        b = Array{Int}(undef, 30)
        @test @allocated(copyto!(b, A)) == 0
        @test b == vcat(A.arrays...)

        A = Vcat(1:10, 1:20)
        @test @inferred(length(A)) == 30
        @test @inferred(A[5]) == A[15] == 5
        @test_throws BoundsError A[31]
        @test reverse(A) == Vcat(reverse(1:20), reverse(1:10))
        b = Array{Int}(undef, 31)
        @test_throws DimensionMismatch copyto!(b, A)
        b = Array{Int}(undef, 30)
        @test @allocated(copyto!(b, A)) == 0
        @test b == vcat(A.arrays...)

        A = Vcat(randn(2,10), randn(4,10))
        @test @inferred(length(A)) == 60
        @test @inferred(size(A)) == (6,10)
        @test_throws BoundsError A[61]
        @test_throws BoundsError A[7,1]
        b = Array{Float64}(undef, 7,10)
        @test_throws DimensionMismatch copyto!(b, A)
        b = Array{Float64}(undef, 6,10)
        @test @allocated(copyto!(b, A)) == 0
        @test b == vcat(A.arrays...)

        @test_throws ArgumentError Vcat()
    end
    @testset "Hcat" begin
        A = Hcat(1:10, 2:11)
        @test_throws BoundsError A[1,3]
        @test @inferred(size(A)) == (10,2)
        @test @inferred(A[5]) == @inferred(A[5,1]) == 5
        @test @inferred(A[11]) == @inferred(A[1,2]) == 2
        b = Array{Int}(undef, 11, 2)
        @test_throws DimensionMismatch copyto!(b, A)
        b = Array{Int}(undef, 10, 2)
        @test @allocated(copyto!(b, A)) == 0
        @test b == hcat(A.arrays...)

        A = Hcat(Vector(1:10), Vector(2:11))
        b = Array{Int}(undef, 10, 2)
        copyto!(b, A)
        @test b == hcat(A.arrays...)
        @test @allocated(copyto!(b, A)) == 0

        A = Hcat(1, zeros(1,5))
        @test A == hcat(1, zeros(1,5))

        A = Hcat(Vector(1:10), randn(10, 2))
        b = Array{Float64}(undef, 10, 3)
        copyto!(b, A)
        @test b == hcat(A.arrays...)
        @test @allocated(copyto!(b, A)) == 0
    end


    @testset "Special pads" begin
        A = Vcat([1,2,3], Zeros(7))
        B = Vcat([1,2], Zeros(8))

        C = @inferred(A+B)
        @test C isa Vcat{Float64,1}
        @test C.arrays[1] isa Vector{Float64}
        @test C.arrays[2] isa Zeros{Float64}
        @test C == Vector(A) + Vector(B)


        B = Vcat([1,2], Ones(8))

        C = @inferred(A+B)
        @test C isa Vcat{Float64,1}
        @test C.arrays[1] isa Vector{Float64}
        @test C.arrays[2] isa Ones{Float64}
        @test C == Vector(A) + Vector(B)

        B = Vcat([1,2], randn(8))

        C = @inferred(A+B)
        @test C isa BroadcastArray{Float64}
        @test C == Vector(A) + Vector(B)

        B = Vcat(SVector(1,2), Ones(8))
        C = @inferred(A+B)
        @test C isa Vcat{Float64,1}
        @test C.arrays[1] isa Vector{Float64}
        @test C.arrays[2] isa Ones{Float64}
        @test C == Vector(A) + Vector(B)


        A = Vcat(SVector(3,4), Zeros(8))
        B = Vcat(SVector(1,2), Ones(8))
        C = @inferred(A+B)
        @test C isa Vcat{Float64,1}
        @test C.arrays[1] isa SVector{2,Int}
        @test C.arrays[2] isa Ones{Float64}
        @test C == Vector(A) + Vector(B)
    end
end


@testset "Kron"  begin
    A = [1,2,3]
    B = [4,5,6,7]

    @test Array(Kron(A)) == A
    K = Kron(A,B)
    @test [K[k] for k=1:length(K)] == Array(K) == kron(A,B)

    A = randn(3)
    K = Kron(A,B)
    @test K isa Kron{Float64}
    @test all(isa.(K.arrays, Vector{Float64}))
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
    @test all(isa.(K.arrays, Matrix{Float64}))
    @test [K[k,j] for k=1:size(K,1), j=1:size(K,2)] == Array(K) == Array(Kron{Float64}(A,B)) == kron(A,B)
end

@testset "BroadcastArray" begin
    A = randn(6,6)
    B = BroadcastArray(exp, A)
    @test Matrix(B) == exp.(A)

    C = BroadcastArray(+, A, 2)
    @test C == A .+ 2
    D = BroadcastArray(+, A, C)
    @test D == A + C

    x = Vcat([3,4], [1,1,1,1,1], 1:3)
    @test x .+ (1:10) isa Vcat
    @test (1:10) .+ x isa Vcat
    @test x + (1:10) isa Vcat
    @test (1:10) + x isa Vcat
    @test x .+ (1:10) == (1:10) .+ x == (1:10) + x == x + (1:10) == Vector(x) + (1:10)

    @test exp.(x) isa Vcat
    @test exp.(x) == exp.(Vector(x))
    @test x .+ 2 isa Vcat
    @test (x .+ 2).arrays[end] ≡ x.arrays[end] .+ 2 ≡ 3:5
    @test x .* 2 isa Vcat
    @test 2 .+ x isa Vcat
    @test 2 .* x isa Vcat
end

@testset "Cache" begin
    A = 1:10
    C = cache(A)
    @test size(C) == (10,)
    @test axes(C) == (Base.OneTo(10),)
    @test all(Vector(C) .=== Vector(A))

    A = reshape(1:10^2, 10,10)
    C = cache(A)
    @test size(C) == (10,10)
    @test axes(C) == (Base.OneTo(10),Base.OneTo(10))
    @test all(Array(C) .=== Array(A))

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
end

@testset "Diff and Cumsum" begin
    x = Vcat([3,4], [1,1,1,1,1], 3)
    y = @inferred(cumsum(x))
    @test y isa Vcat
    @test y == cumsum(Vector(x)) == Cumsum(x)
    @test @inferred(diff(x)) == diff(Vector(x)) == Diff(x)

    @test sum(x) == sum(Vector(x)) == last(y)
    @test cumsum(Vcat(4)) === Vcat(4)

    A = randn(3,4)
    @test Cumsum(A; dims=1) == cumsum(A; dims=1)
    @test Cumsum(A; dims=2) == cumsum(A; dims=2)
    @test Diff(A; dims=1) == diff(A; dims=1)
    @test Diff(A; dims=2) == diff(A; dims=2)
end



@testset "broadcast Vcat" begin
    x = Vcat(1:2, [1,1,1,1,1], 3)
    y = 1:8
    f = (x,y) -> cos(x*y)
    @test f.(x,y) isa Vcat
    @test @inferred(broadcast(f,x,y)) == f.(Vector(x), Vector(y))

    @test (x .+ y) isa Vcat
    @test (x .+ y).arrays[1] isa AbstractRange
    @test (x .+ y).arrays[end] isa Int

    z = Vcat(1:2, [1,1,1,1,1], 3)
    (x .+ z)  isa BroadcastArray
    (x + z) isa BroadcastArray
    @test Vector( x .+ z) == Vector( x + z) == Vector(x) + Vector(z)

    # Lazy mixed with Static treats as Lazy
    s = SVector(1,2,3,4,5,6,7,8)
    @test f.(x , s) isa Vcat
    @test f.(x , s) == f.(Vector(x), Vector(s))

    # these are special cased
    @test Vcat(1, Ones(5))  + Vcat(2, Fill(2.0,5)) ≡ Vcat(3, Fill(3.0,5))
    @test Vcat(SVector(1,2,3), Ones(5))  + Vcat(SVector(4,5,6), Fill(2.0,5)) ≡
        Vcat(SVector(5,7,9), Fill(3.0,5))
end

@testset "maximum/minimum Vcat" begin
    x = Vcat(1:2, [1,1,1,1,1], 3)
    @test maximum(x) == 3
    @test minimum(x) == 1
end
