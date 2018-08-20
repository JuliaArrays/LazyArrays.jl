using Test, LinearAlgebra, LazyArrays


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

@testset "gemv adjtrans" begin
    for A in (randn(5,5), view(randn(5,5),:,:), view(randn(5,5),1:5,:),
              view(randn(5,5),1:5,1:5), view(randn(5,5),:,1:5)),
                b in (randn(5), view(randn(5),:), view(randn(5),1:5), view(randn(9),1:2:9)),
                Ac in (transpose(A), A')
        c = similar(b);

        c .= Mul(Ac,b)
        @test all(c .=== BLAS.gemv!('T', 1.0, A, b, 0.0, similar(c)))

        b .= Mul(Ac,b)
        @test all(c .=== b)

        c .= 2.0 .* Mul(Ac,b)
        @test all(c .=== BLAS.gemv!('T', 2.0, A, b, 0.0, similar(c)))

        c = copy(b)
        c .= Mul(Ac,b) .+ c
        @test all(c .=== BLAS.gemv!('T', 1.0, A, b, 1.0, copy(b)))


        c = copy(b)
        c .= Mul(Ac,b) .+ 2.0 .* c
        @test all(c .=== BLAS.gemv!('T', 1.0, A, b, 2.0, copy(b)))

        c = copy(b)
        c .= 2.0 .* Mul(Ac,b) .+ c
        @test all(c .=== BLAS.gemv!('T', 2.0, A, b, 1.0, copy(b)))


        c = copy(b)
        c .= 3.0 .* Mul(Ac,b) .+ 2.0 .* c
        @test all(c .=== BLAS.gemv!('T', 3.0, A, b, 2.0, copy(b)))

        d = similar(c)
        c = copy(b)
        d .= 3.0 .* Mul(Ac,b) .+ 2.0 .* c
        @test all(d .=== BLAS.gemv!('T', 3.0, A, b, 2.0, copy(b)))
    end

    # test mixed types
    let (A, b, c) = (randn(5,5), randn(5), 1.0:5.0)
        for Ac in (transpose(A), A')
            d = similar(b)
            d .= Mul(Ac,b) .+ c
            @test all(d .=== BLAS.gemv!('T', 1.0, A, b, 1.0, Vector{Float64}(c)))

            d .= Mul(Ac,b) .+ 2.0 .* c
            @test all(d .=== BLAS.gemv!('T', 1.0, A, b, 2.0, Vector{Float64}(c)))

            d .= 2.0 .* Mul(Ac,b) .+ c
            @test all(d .=== BLAS.gemv!('T', 2.0, A, b, 1.0, Vector{Float64}(c)))

            d .= 3.0 .* Mul(Ac,b) .+ 2.0 .* c
            @test all(d .=== BLAS.gemv!('T', 3.0, A, b, 2.0, Vector{Float64}(c)))
        end
    end
end

@testset "gemm adjtrans" begin
    for A in (randn(5,5), view(randn(5,5),:,:)),
        B in (randn(5,5), view(randn(5,5),1:5,:))
        for Ac in (transpose(A), A')
            C = similar(B)
            C .= Mul(Ac,B)
            @test all(C .=== BLAS.gemm!('T', 'N', 1.0, A, B, 0.0, similar(C)))

            B .= Mul(Ac,B)
            @test all(C .=== B)

            C .= 2.0 .* Mul(Ac,B)
            @test all(C .=== BLAS.gemm!('T', 'N', 2.0, A, B, 0.0, similar(C)))

            C = copy(B)
            C .= Mul(Ac,B) .+ C
            @test all(C .=== BLAS.gemm!('T', 'N', 1.0, A, B, 1.0, copy(B)))


            C = copy(B)
            C .= Mul(Ac,B) .+ 2.0 .* C
            @test all(C .=== BLAS.gemm!('T', 'N', 1.0, A, B, 2.0, copy(B)))

            C = copy(B)
            C .= 2.0 .* Mul(Ac,B) .+ C
            @test all(C .=== BLAS.gemm!('T', 'N', 2.0, A, B, 1.0, copy(B)))


            C = copy(B)
            C .= 3.0 .* Mul(Ac,B) .+ 2.0 .* C
            @test all(C .=== BLAS.gemm!('T', 'N', 3.0, A, B, 2.0, copy(B)))

            d = similar(C)
            C = copy(B)
            d .= 3.0 .* Mul(Ac,B) .+ 2.0 .* C
            @test all(d .=== BLAS.gemm!('T', 'N', 3.0, A, B, 2.0, copy(B)))
        end
        for Bc in (transpose(B), B')
            C = similar(B)
            C .= Mul(A,Bc)
            @test all(C .=== BLAS.gemm!('N', 'T', 1.0, A, B, 0.0, similar(C)))

            B .= Mul(A,Bc)
            @test all(C .=== B)

            C .= 2.0 .* Mul(A,Bc)
            @test all(C .=== BLAS.gemm!('N', 'T', 2.0, A, B, 0.0, similar(C)))

            C = copy(B)
            C .= Mul(A,Bc) .+ C
            @test all(C .=== BLAS.gemm!('N', 'T', 1.0, A, B, 1.0, copy(B)))


            C = copy(B)
            C .= Mul(A,Bc) .+ 2.0 .* C
            @test all(C .=== BLAS.gemm!('N', 'T', 1.0, A, B, 2.0, copy(B)))

            C = copy(B)
            C .= 2.0 .* Mul(A,Bc) .+ C
            @test all(C .=== BLAS.gemm!('N', 'T', 2.0, A, B, 1.0, copy(B)))


            C = copy(B)
            C .= 3.0 .* Mul(A,Bc) .+ 2.0 .* C
            @test all(C .=== BLAS.gemm!('N', 'T', 3.0, A, B, 2.0, copy(B)))

            d = similar(C)
            C = copy(B)
            d .= 3.0 .* Mul(A,Bc) .+ 2.0 .* C
            @test all(d .=== BLAS.gemm!('N', 'T', 3.0, A, B, 2.0, copy(B)))
        end
        for Ac in (transpose(A), A'), Bc in (transpose(B), B')
            C = similar(B)
            C .= Mul(Ac,Bc)
            @test all(C .=== BLAS.gemm!('T', 'T', 1.0, A, B, 0.0, similar(C)))

            B .= Mul(Ac,Bc)
            @test all(C .=== B)

            C .= 2.0 .* Mul(Ac,Bc)
            @test all(C .=== BLAS.gemm!('T', 'T', 2.0, A, B, 0.0, similar(C)))

            C = copy(B)
            C .= Mul(Ac,Bc) .+ C
            @test all(C .=== BLAS.gemm!('T', 'T', 1.0, A, B, 1.0, copy(B)))


            C = copy(B)
            C .= Mul(Ac,Bc) .+ 2.0 .* C
            @test all(C .=== BLAS.gemm!('T', 'T', 1.0, A, B, 2.0, copy(B)))

            C = copy(B)
            C .= 2.0 .* Mul(Ac,Bc) .+ C
            @test all(C .=== BLAS.gemm!('T', 'T', 2.0, A, B, 1.0, copy(B)))


            C = copy(B)
            C .= 3.0 .* Mul(Ac,Bc) .+ 2.0 .* C
            @test all(C .=== BLAS.gemm!('T', 'T', 3.0, A, B, 2.0, copy(B)))

            d = similar(C)
            C = copy(B)
            d .= 3.0 .* Mul(Ac,Bc) .+ 2.0 .* C
            @test all(d .=== BLAS.gemm!('T', 'T', 3.0, A, B, 2.0, copy(B)))
        end
    end
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
    Ac = A'
    blasnoalloc(c, 2.0, Ac, x, 3.0, y)
    @test @allocated(blasnoalloc(c, 2.0, Ac, x, 3.0, y)) == 0
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

@testset "Ldiv" begin
    A = randn(5,5)
    b = randn(5)
    typeof(Ldiv(A,b))
    @test all(copyto!(similar(b), Ldiv(A,b)) .== (A\b))
end
