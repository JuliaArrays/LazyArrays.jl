using Test, LinearAlgebra, LazyArrays, BenchmarkTools


@testset "gemv" begin
    for A in (randn(5,5), view(randn(5,5),:,:), view(randn(5,5),1:5,:),
              view(randn(5,5),1:5,1:5), view(randn(5,5),:,1:5)),
        b in (randn(5), view(randn(5),:), view(randn(5),1:5), view(randn(9),1:2:9))
        c = similar(b);

        c .= Mul(A,b)
        @test all(c .=== BLAS.gemv!('N', 1.0, A, b, 0.0, similar(c)))

        b .= Mul(A,b)
        @test all(c .=== b)

        c .= 2.0 .* Mul(A,b)
        @test all(c .=== BLAS.gemv!('N', 2.0, A, b, 0.0, similar(c)))

        c = copy(b)
        c .= Mul(A,b) .+ c
        @test all(c .=== BLAS.gemv!('N', 1.0, A, b, 1.0, copy(b)))


        c = copy(b)
        c .= Mul(A,b) .+ 2.0 .* c
        @test all(c .=== BLAS.gemv!('N', 1.0, A, b, 2.0, copy(b)))

        c = copy(b)
        c .= 2.0 .* Mul(A,b) .+ c
        @test all(c .=== BLAS.gemv!('N', 2.0, A, b, 1.0, copy(b)))


        c = copy(b)
        c .= 3.0 .* Mul(A,b) .+ 2.0 .* c
        @test all(c .=== BLAS.gemv!('N', 3.0, A, b, 2.0, copy(b)))

        d = similar(c)
        c = copy(b)
        d .= 3.0 .* Mul(A,b) .+ 2.0 .* c
        @test all(d .=== BLAS.gemv!('N', 3.0, A, b, 2.0, copy(b)))
    end

    # test mixed types
    let (A, b, c) = (randn(5,5), randn(5), 1.0:5.0)
        d = similar(b)
        d .= Mul(A,b) .+ c
        @test all(d .=== BLAS.gemv!('N', 1.0, A, b, 1.0, Vector{Float64}(c)))

        d .= Mul(A,b) .+ 2.0 .* c
        @test all(d .=== BLAS.gemv!('N', 1.0, A, b, 2.0, Vector{Float64}(c)))

        d .= 2.0 .* Mul(A,b) .+ c
        @test all(d .=== BLAS.gemv!('N', 2.0, A, b, 1.0, Vector{Float64}(c)))

        d .= 3.0 .* Mul(A,b) .+ 2.0 .* c
        @test all(d .=== BLAS.gemv!('N', 3.0, A, b, 2.0, Vector{Float64}(c)))
    end
end

@testset "gemm" begin
    for A in (randn(5,5), view(randn(5,5),:,:), view(randn(5,5),1:5,:),
              view(randn(5,5),1:5,1:5), view(randn(5,5),:,1:5)),
        B in (randn(5,5), view(randn(5,5),:,:), view(randn(5,5),1:5,:),
                  view(randn(5,5),1:5,1:5), view(randn(5,5),:,1:5))
        C = similar(B);

        C .= Mul(A,B)
        @test all(C .=== BLAS.gemm!('N', 'N', 1.0, A, B, 0.0, similar(C)))

        B .= Mul(A,B)
        @test all(C .=== B)

        C .= 2.0 .* Mul(A,B)
        @test all(C .=== BLAS.gemm!('N', 'N', 2.0, A, B, 0.0, similar(C)))

        C = copy(B)
        C .= Mul(A,B) .+ C
        @test all(C .=== BLAS.gemm!('N', 'N', 1.0, A, B, 1.0, copy(B)))


        C = copy(B)
        C .= Mul(A,B) .+ 2.0 .* C
        @test all(C .=== BLAS.gemm!('N', 'N', 1.0, A, B, 2.0, copy(B)))

        C = copy(B)
        C .= 2.0 .* Mul(A,B) .+ C
        @test all(C .=== BLAS.gemm!('N', 'N', 2.0, A, B, 1.0, copy(B)))


        C = copy(B)
        C .= 3.0 .* Mul(A,B) .+ 2.0 .* C
        @test all(C .=== BLAS.gemm!('N', 'N', 3.0, A, B, 2.0, copy(B)))

        d = similar(C)
        C = copy(B)
        d .= 3.0 .* Mul(A,B) .+ 2.0 .* C
        @test all(d .=== BLAS.gemm!('N', 'N', 3.0, A, B, 2.0, copy(B)))
    end
end

@testset "adjtrans" begin
    A = randn(5,5); b = randn(5);
    c = copy(b)
    c .= 3.0 .* Mul(transpose(A),b) .+ 2.0 .* c
    @test all(c .=== BLAS.gemv!('T', 3.0, A, b, 2.0, copy(b)))
end

@testset "no allocation" begin
    function blasnoalloc(c, α, A, x, β, y)
        c .= Mul(A,x)
        c .= α .* Mul(A,x)
        c .= Mul(A,x) .+ y
        c .= α .* Mul(A,x) .+ y
        c .= Mul(A,x) .+ β .* y
        c .= α .* Mul(A,x) .+ β .* y
    end

    A = randn(5,5); x = randn(5); y = randn(5); c = similar(y);
    blasnoalloc(c, 2.0, A, x, 3.0, y)
    @test @allocated(blasnoalloc(c, 2.0, A, x, 3.0, y)) == 0
end



@testset "concat" begin
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
    @test @belapsed(copyto!($b,$A)) < @belapsed(vcat($A.arrays...))

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
    @test @belapsed(copyto!($b,$A)) < @belapsed(vcat($A.arrays...))

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
    @test @belapsed(copyto!($b,$A)) < @belapsed(vcat($A.arrays...))

    A = Hcat(1:10, 2:11)
    @test @inferred(size(A)) == (10,2)
    @test @inferred(A[5]) == @inferred(A[5,1]) == 5
    @test @inferred(A[11]) == @inferred(A[1,2]) == 2
    b = Array{Int}(undef, 11, 2)
    @test_throws DimensionMismatch copyto!(b, A)
    b = Array{Int}(undef, 10, 2)
    @test @allocated(copyto!(b, A)) == 0
    @test b == hcat(A.arrays...)
    @test @belapsed(copyto!($b,$A)) < @belapsed(hcat($A.arrays...))

    A = Hcat(1, zeros(1,5))
    @test A == hcat(1, zeros(1,5))
end


@testset "BroadcastArray" begin
    A = randn(6,6)
    B = BroadcastArray(exp, A)
    @test Matrix(B) == exp.(A)

    B = BroadcastArray(+, A, 2)
    @test B == A .+ 2
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
