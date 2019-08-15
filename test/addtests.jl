using LazyArrays, Test
import LazyArrays: Add, AddArray, MulAdd, materialize!


@testset "Add" begin
    @testset "Mul-Add" begin
        A = Add(randn(5,5), randn(5,5))
        b = randn(5)
        c = similar(b)
        fill!(c,NaN)
        @test (c .= @~ A*b) ≈ A.args[1]*b + A.args[2]*b
    end

    @testset "gemv Float64" begin
        for A in (AddArray(randn(5,5), randn(5,5)),
                  AddArray(randn(5,5), view(randn(9, 5), 1:2:9, :))),
            b in (randn(5), view(randn(5),:), view(randn(5),1:5), view(randn(9),1:2:9))

            Ã = copy(A)
            c = similar(b)
            fill!(c,NaN)

            materialize!(MulAdd(1.0, A, b, 0.0, c))
            @test c ≈ Ã*b ≈ BLAS.gemv!('N', 1.0, Ã, b, 0.0, similar(c))

            c = similar(b)
            fill!(c,NaN)
            c .= @~ A*b
            @test c ≈ Ã*b ≈ BLAS.gemv!('N', 1.0, Ã, b, 0.0, similar(c))

            copyto!(c, @~ A*b)
            @test c ≈ Ã*b ≈ BLAS.gemv!('N', 1.0, Ã, b, 0.0, similar(c))

            b̃ = copy(b)
            copyto!(b̃, Mul(A,b̃))
            @test c ≈ b̃

            c .= @~ 2.0 * A * b
            @test c ≈ BLAS.gemv!('N', 2.0, Ã, b, 0.0, similar(c))

            c = copy(b)
            c .= @~ A*b + c
            @test c ≈ BLAS.gemv!('N', 1.0, Ã, b, 1.0, copy(b))

            c = copy(b)
            c .= @~ A*b + 2.0 * c
            @test c ≈ BLAS.gemv!('N', 1.0, Ã, b, 2.0, copy(b))

            c = copy(b)
            c .= @~ 2.0 * A*b + c
            @test c ≈ BLAS.gemv!('N', 2.0, Ã, b, 1.0, copy(b))

            c = copy(b)
            c .= @~ 3.0 * A*b + 2.0 * c
            @test c ≈ BLAS.gemv!('N', 3.0, Ã, b, 2.0, copy(b))

            d = similar(c)
            c = copy(b)
            d .= @~ 3.0 * A*b + 2.0 * c
            @test d ≈ BLAS.gemv!('N', 3.0, Ã, b, 2.0, copy(b))
        end
    end

    @testset "gemm" begin
        for A in (AddArray(randn(5,5), randn(5,5)),
                  AddArray(randn(5,5), view(randn(9, 5), 1:2:9, :))),
            B in (randn(5,5), view(randn(5,5),:,:), view(randn(5,5),1:5,:),
                  view(randn(5,5),1:5,1:5), view(randn(5,5),:,1:5))

            Ã = copy(A)
            C = similar(B)

            C .= @~ A*B
            @test C ≈ BLAS.gemm!('N', 'N', 1.0, Ã, B, 0.0, similar(C))

            B .= @~ A*B
            @test C ≈ B

            C .= @~ 2.0 * A*B
            @test C ≈ BLAS.gemm!('N', 'N', 2.0, Ã, B, 0.0, similar(C))

            C = copy(B)
            C .= @~ A*B + C
            @test C ≈ BLAS.gemm!('N', 'N', 1.0, Ã, B, 1.0, copy(B))


            C = copy(B)
            C .= @~ A*B + 2.0 * C
            @test C ≈ BLAS.gemm!('N', 'N', 1.0, Ã, B, 2.0, copy(B))

            C = copy(B)
            C .= @~ 2.0 * A*B + C
            @test C ≈ BLAS.gemm!('N', 'N', 2.0, Ã, B, 1.0, copy(B))


            C = copy(B)
            C .= @~ 3.0 * A*B + 2.0 * C
            @test C ≈ BLAS.gemm!('N', 'N', 3.0, Ã, B, 2.0, copy(B))

            d = similar(C)
            C = copy(B)
            d .= @~ 3.0 * A*B + 2.0 * C
            @test d ≈ BLAS.gemm!('N', 'N', 3.0, Ã, B, 2.0, copy(B))
        end
    end
end
