using Test, LinearAlgebra, LazyLinearAlgebra


@testset "gemv" begin
    for A in (randn(5,5), view(randn(5,5),:,:), view(randn(5,5),1:5,:),
              view(randn(5,5),1:5,1:5), view(randn(5,5),:,1:5))
        b = randn(5); c = similar(b);

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
        c .= 3.0 .* Mul(A,b) .+ 2.0 .* c
        @test all(c .=== BLAS.gemv!('N', 3.0, A, b, 2.0, copy(b)))

        d = similar(c)
        c = copy(b)
        d .= 3.0 .* Mul(A,b) .+ 2.0 .* c
        @test all(d .=== BLAS.gemv!('N', 3.0, A, b, 2.0, copy(b)))
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
