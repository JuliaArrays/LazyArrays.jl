using Test, LinearAlgebra, LazyArrays, StaticArrays, FillArrays
import LazyArrays: CachedArray, colsupport, rowsupport

include("memorylayouttests.jl")
include("applytests.jl")
include("multests.jl")
include("ldivtests.jl")
include("addtests.jl")
include("setoptests.jl")
include("macrotests.jl")
include("lazymultests.jl")

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
        @test copy(A) isa Vcat
        @test copy(A) == A
        @test copy(A) !== A
        @test vec(A) === A
        @test A' == transpose(A) == Vector(A)'

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
        @test copy(A) === A
        @test vec(A) === A
        @test A' == transpose(A) == Vector(A)'
        @test A' === Hcat((1:10)', (1:20)')
        @test transpose(A) === Hcat(transpose(1:10), transpose(1:20))

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
        @test copy(A) isa Vcat
        @test copy(A) == A
        @test copy(A) !== A
        @test vec(A) == vec(Matrix(A))
        @test A' == transpose(A) == Matrix(A)'

        A = Vcat(randn(2,10).+im.*randn(2,10), randn(4,10).+im.*randn(4,10))
        @test eltype(A) == ComplexF64
        @test @inferred(length(A)) == 60
        @test @inferred(size(A)) == (6,10)
        @test_throws BoundsError A[61]
        @test_throws BoundsError A[7,1]
        b = Array{ComplexF64}(undef, 7,10)
        @test_throws DimensionMismatch copyto!(b, A)
        b = Array{ComplexF64}(undef, 6,10)
        @test @allocated(copyto!(b, A)) == 0
        @test b == vcat(A.arrays...)
        @test copy(A) isa Vcat
        @test copy(A) == A
        @test copy(A) !== A
        @test vec(A) == vec(Matrix(A))
        @test A' == Matrix(A)'
        @test transpose(A) == transpose(Matrix(A))

        @test Vcat() isa Vcat{Any,1,Tuple{}}

        A = Vcat(1,zeros(3,1))
        @test_broken A isa AbstractMatrix
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
        @test copy(A) === A
        @test vec(A) == vec(Matrix(A))
        @test vec(A) === Vcat(1:10,2:11)
        @test A' == Matrix(A)'
        @test A' === Vcat((1:10)', (2:11)')

        A = Hcat(Vector(1:10), Vector(2:11))
        b = Array{Int}(undef, 10, 2)
        copyto!(b, A)
        @test b == hcat(A.arrays...)
        @test @allocated(copyto!(b, A)) == 0
        @test copy(A) isa Hcat
        @test copy(A) == A
        @test copy(A) !== A
        @test vec(A) == vec(Matrix(A))
        @test vec(A) === Vcat(A.arrays...)
        @test A' == Matrix(A)'

        A = Hcat(1, zeros(1,5))
        @test A == hcat(1, zeros(1,5))
        @test vec(A) == vec(Matrix(A))
        @test_broken A' == Matrix(A)'

        A = Hcat(Vector(1:10), randn(10, 2))
        b = Array{Float64}(undef, 10, 3)
        copyto!(b, A)
        @test b == hcat(A.arrays...)
        @test @allocated(copyto!(b, A)) == 0
        @test vec(A) == vec(Matrix(A))

        A = Hcat(randn(5).+im.*randn(5), randn(5,2).+im.*randn(5,2))
        b = Array{ComplexF64}(undef, 5, 3)
        copyto!(b, A)
        @test b == hcat(A.arrays...)
        @test @allocated(copyto!(b, A)) == 0
        @test vec(A) == vec(Matrix(A))
        @test A' == Matrix(A)'
        @test transpose(A) == transpose(Matrix(A))
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

    @testset "Empty Vcat" begin
        @test @inferred(Vcat{Int}([1])) == [1]        
        @test @inferred(Vcat{Int}(())) == @inferred(Vcat{Int}()) == Int[]        
    end

    @testset "in" begin
        @test 1 in Vcat(1, 1:10_000_000_000)
        @test 100_000_000 in Vcat(1, 1:10_000_000_000)
    end

    @testset "convert" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            Z = Vcat(zero(T),Zeros{T}(10))
            @test convert(AbstractArray,Z) ≡ AbstractArray(Z) ≡ Z
            @test convert(AbstractArray{T},Z) ≡ AbstractArray{T}(Z) ≡ Z
            @test convert(AbstractVector{T},Z) ≡ AbstractVector{T}(Z) ≡ Z
        end
    end

    @testset "setindex!" begin
        x = randn(5)
        y = randn(6)
        A = Vcat(x, y, 3)
        A[1] = 1
        @test A[1] == x[1] == 1
        A[6] = 2
        @test A[6] == y[1] == 2
        @test_throws MethodError A[12] = 3
        @test_throws BoundsError A[13] = 3

        x = randn(2,2); y = randn(3,2)
        A = Vcat(x,y)
        A[1,1] = 1
        @test A[1,1] == x[1,1] == 1
        A[3,1] = 2
        @test A[3,1] == y[1,1] == 2
        A[6] = 3
        @test A[1,2] == x[1,2] == 3

        x = randn(2,2); y = randn(2,3)
        B = Hcat(x,y)
        B[1,1] = 1
        @test B[1,1] == x[1,1] == 1
        B[1,3] = 2
        @test B[1,3] == y[1,1] == 2
    end

    @testset "fill!" begin
        A = Vcat([1,2,3],[4,5,6])
        fill!(A,2)
        @test A == fill(2,6)

        A = Vcat(2,[4,5,6])
        @test fill!(A,2) == fill(2,4)
        @test_throws ArgumentError fill!(A,3)

        A = Hcat([1,2,3],[4,5,6])
        fill!(A,2)
        @test A == fill(2,3,2)
    end

    @testset "Any/All" begin
        @test all(Vcat(true, Fill(true,100_000_000)))
        @test any(Vcat(false, Fill(true,100_000_000)))
        @test all(iseven, Vcat(2, Fill(4,100_000_000)))
        @test any(iseven, Vcat(2, Fill(1,100_000_000)))
        @test_throws TypeError all(Vcat(1))
        @test_throws TypeError any(Vcat(1))
    end

    @testset "isbitsunion #45" begin 
        @test copyto!(Vector{Vector{Int}}(undef,6), Vcat([[1], [2], [3]], [[1], [2], [3]])) ==
            [[1], [2], [3], [1], [2], [3]]

        a = Vcat{Union{Float64,UInt8}}([1.0], [UInt8(1)])
        @test Base.isbitsunion(eltype(a))
        r = Vector{Union{Float64,UInt8}}(undef,2)
        @test copyto!(r, a) == a
        @test r == a
        @test copyto!(Vector{Float64}(undef,2), a) == [1.0,1.0]
    end
end


@testset "Kron"  begin
    A = [1,2,3]
    B = [4,5,6,7]

    @test Array(@inferred(Kron(A))) == A
    K = @inferred(Kron(A,B))
    @test [K[k] for k=1:length(K)] == Array(K) == kron(A,B)

    A = randn(3)
    K = @inferred(Kron(A,B))
    @test K isa Kron{Float64}
    @test all(K.arrays .=== (A,B))
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
    @test all(K.arrays .=== (A,B))
    @test [K[k,j] for k=1:size(K,1), j=1:size(K,2)] == Array(K) == Array(Kron{Float64}(A,B)) == kron(A,B)

    K = @inferred(Kron{Float64}(Eye{Float64}(1), zeros(4)))
    @test Array(K) == zeros(4,1)
end

@testset "BroadcastArray" begin
    A = randn(6,6)
    B = BroadcastArray(exp, A)
    @test Matrix(B) == exp.(A)

    C = BroadcastArray(+, A, 2)
    @test C == A .+ 2
    D = BroadcastArray(+, A, C)
    @test D == A + C

    @test sum(B) ≈ sum(exp, A)
    @test sum(C) ≈ sum(A .+ 2)

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

    A = Vcat([[1 2; 3 4]], [[4 5; 6 7]])
    @test A .+ Ref(I) == Ref(I) .+ A == Vcat([[2 2; 3 5]], [[5 5; 6 8]])
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

    A = collect(1:5)
    C = cache(A)
    @test C isa Vector{Int}
    C[1] = 2
    @test A[1] ≠ 2
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
    @test (x .+ z) isa BroadcastArray
    @test (x + z) isa BroadcastArray
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

@testset "vector*matrix broadcasting #27" begin
    H = [1., 0.]
    @test Mul(H, H') .+ 1 == H*H' .+ 1
    B =  randn(2,2)
    @test Mul(H, H') .+ B == H*H' .+ B
end

@testset "col/rowsupport" begin
    A = randn(5,6)
    @test rowsupport(A,1) === Base.OneTo(6)
    @test colsupport(A,1) === Base.OneTo(5)
    D = Diagonal(randn(5))
    @test rowsupport(D,3) === colsupport(D,3) === 3:3
    Z = Zeros(5)
    @test rowsupport(Z,1) === colsupport(Z,1) === 1:0
end