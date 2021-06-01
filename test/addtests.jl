using LazyArrays, Test
import LazyArrays: Add, AddArray, MulAdd, materialize!, MemoryLayout, ApplyLayout

@testset "Add/Subtract" begin
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

                Ã = Array(A)
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

                b̃ = Array(b)
                copyto!(b̃, Mul(A,b̃))
                @test c ≈ b̃

                c .= @~ 2.0 * A * b
                @test c ≈ BLAS.gemv!('N', 2.0, Ã, b, 0.0, similar(c))

                c = Array(b)
                c .= @~ A*b + c
                @test c ≈ BLAS.gemv!('N', 1.0, Ã, b, 1.0, Array(b))

                c = Array(b)
                c .= @~ A*b + 2.0 * c
                @test c ≈ BLAS.gemv!('N', 1.0, Ã, b, 2.0, Array(b))

                c = Array(b)
                c .= @~ 2.0 * A*b + c
                @test c ≈ BLAS.gemv!('N', 2.0, Ã, b, 1.0, Array(b))

                c = Array(b)
                c .= @~ 3.0 * A*b + 2.0 * c
                @test c ≈ BLAS.gemv!('N', 3.0, Ã, b, 2.0, Array(b))

                d = similar(c)
                c = Array(b)
                d .= @~ 3.0 * A*b + 2.0 * c
                @test d ≈ BLAS.gemv!('N', 3.0, Ã, b, 2.0, Array(b))
            end
        end

        @testset "gemm" begin
            for A in (AddArray(randn(5,5), randn(5,5)),
                    AddArray(randn(5,5), view(randn(9, 5), 1:2:9, :))),
                B in (randn(5,5), view(randn(5,5),:,:), view(randn(5,5),1:5,:),
                    view(randn(5,5),1:5,1:5), view(randn(5,5),:,1:5))

                Ã = Array(A)
                C = similar(B)

                C .= @~ A*B
                @test C ≈ BLAS.gemm!('N', 'N', 1.0, Ã, B, 0.0, similar(C))

                B .= @~ A*B
                @test C ≈ B

                C .= @~ 2.0 * A*B
                @test C ≈ BLAS.gemm!('N', 'N', 2.0, Ã, B, 0.0, similar(C))

                C = Array(B)
                C .= @~ A*B + C
                @test C ≈ BLAS.gemm!('N', 'N', 1.0, Ã, B, 1.0, Array(B))


                C = Array(B)
                C .= @~ A*B + 2.0 * C
                @test C ≈ BLAS.gemm!('N', 'N', 1.0, Ã, B, 2.0, Array(B))

                C = Array(B)
                C .= @~ 2.0 * A*B + C
                @test C ≈ BLAS.gemm!('N', 'N', 2.0, Ã, B, 1.0, Array(B))


                C = Array(B)
                C .= @~ 3.0 * A*B + 2.0 * C
                @test C ≈ BLAS.gemm!('N', 'N', 3.0, Ã, B, 2.0, Array(B))

                d = similar(C)
                C = Array(B)
                d .= @~ 3.0 * A*B + 2.0 * C
                @test d ≈ BLAS.gemm!('N', 'N', 3.0, Ã, B, 2.0, Array(B))
            end
        end
    end

    @testset "Subtract" begin
        @testset "Mul-Subtract" begin
            A = applied(-, randn(5,5), randn(5,5))
            b = randn(5)
            c = similar(b)
            fill!(c,NaN)
            @test (c .= @~ A*b) ≈ A.args[1]*b - A.args[2]*b
        end

        @testset "gemv Float64" begin
            for A in (ApplyArray(-, randn(5,5), randn(5,5)),
                    ApplyArray(-, randn(5,5), view(randn(9, 5), 1:2:9, :))),
                b in (randn(5), view(randn(5),:), view(randn(5),1:5), view(randn(9),1:2:9))

                Ã = Array(A)
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

                b̃ = Array(b)
                copyto!(b̃, Mul(A,b̃))
                @test c ≈ b̃

                c .= @~ 2.0 * A * b
                @test c ≈ BLAS.gemv!('N', 2.0, Ã, b, 0.0, similar(c))

                c = Array(b)
                c .= @~ A*b + c
                @test c ≈ BLAS.gemv!('N', 1.0, Ã, b, 1.0, Array(b))

                c = Array(b)
                c .= @~ A*b + 2.0 * c
                @test c ≈ BLAS.gemv!('N', 1.0, Ã, b, 2.0, Array(b))

                c = Array(b)
                c .= @~ 2.0 * A*b + c
                @test c ≈ BLAS.gemv!('N', 2.0, Ã, b, 1.0, Array(b))

                c = Array(b)
                c .= @~ 3.0 * A*b + 2.0 * c
                @test c ≈ BLAS.gemv!('N', 3.0, Ã, b, 2.0, Array(b))

                d = similar(c)
                c = Array(b)
                d .= @~ 3.0 * A*b + 2.0 * c
                @test d ≈ BLAS.gemv!('N', 3.0, Ã, b, 2.0, Array(b))
            end
        end

        @testset "gemm" begin
            for A in (ApplyArray(-, randn(5,5), randn(5,5)),
                    ApplyArray(-, randn(5,5), view(randn(9, 5), 1:2:9, :))),
                B in (randn(5,5), view(randn(5,5),:,:), view(randn(5,5),1:5,:),
                    view(randn(5,5),1:5,1:5), view(randn(5,5),:,1:5))

                Ã = Array(A)
                C = similar(B)

                C .= @~ A*B
                @test C ≈ BLAS.gemm!('N', 'N', 1.0, Ã, B, 0.0, similar(C))

                B .= @~ A*B
                @test C ≈ B

                C .= @~ 2.0 * A*B
                @test C ≈ BLAS.gemm!('N', 'N', 2.0, Ã, B, 0.0, similar(C))

                C = Array(B)
                C .= @~ A*B + C
                @test C ≈ BLAS.gemm!('N', 'N', 1.0, Ã, B, 1.0, Array(B))


                C = Array(B)
                C .= @~ A*B + 2.0 * C
                @test C ≈ BLAS.gemm!('N', 'N', 1.0, Ã, B, 2.0, Array(B))

                C = Array(B)
                C .= @~ 2.0 * A*B + C
                @test C ≈ BLAS.gemm!('N', 'N', 2.0, Ã, B, 1.0, Array(B))


                C = Array(B)
                C .= @~ 3.0 * A*B + 2.0 * C
                @test C ≈ BLAS.gemm!('N', 'N', 3.0, Ã, B, 2.0, Array(B))

                d = similar(C)
                C = Array(B)
                d .= @~ 3.0 * A*B + 2.0 * C
                @test d ≈ BLAS.gemm!('N', 'N', 3.0, Ã, B, 2.0, Array(B))
            end
        end
    end


    @testset "view" begin
        A = ApplyArray(+, randn(5,5), randn(5,5))
        Ã = Array(A)
        V = view(A,2:3,2:4)
        @test MemoryLayout(typeof(V)) == ApplyLayout{typeof(+)}()
        b = randn(3)
        c = randn(2)
        materialize!(MulAdd(1.0, V, b, 0.0, c))
        @test c ≈ view(Ã,2:3,2:4)*b

        A = ApplyArray(-, randn(5,5), randn(5,5))
        Ã = Array(A)
        V = view(A,2:3,2:4)
        @test MemoryLayout(typeof(V)) == ApplyLayout{typeof(-)}()
        b = randn(3)
        c = randn(2)
        materialize!(MulAdd(1.0, V, b, 0.0, c))
        @test c ≈ view(Ã,2:3,2:4)*b
    end

    @testset "Broadcast add" begin
        a = randn(5)
        ã = reshape(a,5,1)
        b = randn(5)
        A = randn(5,5)
        c = [2.3]

        @testset "+" begin
            B = BroadcastArray(+, a, A)
            @test @inferred(B * b) isa Vector
            @test @inferred(B * A) isa Matrix
            @test @inferred(A * B) isa Matrix
            @test B * B isa Matrix
            @test B * b ≈ (a .+ A) * b
            @test B * A ≈ (a .+ A) * A
            @test A * B ≈ A * (a .+ A)
            @test B * B ≈ (a .+ A) * (a .+ A)
            B = BroadcastArray(+, A, a)
            @test B * b ≈ (a .+ A) * b
            @test B * A ≈ (a .+ A) * A
            @test A * B ≈ A * (a .+ A)
            B = BroadcastArray(+, A, A)
            @test B * b ≈ (A .+ A) * b
            @test B * A ≈ (A .+ A) * A
            @test A * B ≈ A * (A .+ A)
            B = BroadcastArray(+, ã, A)
            @test B * b ≈ (ã .+ A) * b
            @test B * A ≈ (ã .+ A) * A
            @test A * B ≈ A * (ã .+ A)
            B = BroadcastArray(+, A, ã)
            @test B * b ≈ (ã .+ A) * b
            @test B * A ≈ (ã .+ A) * A
            @test A * B ≈ A * (ã .+ A)
            B = BroadcastArray(+, a', A)
            @test B * b ≈ (a' .+ A) * b
            @test B * A ≈ (a' .+ A) * A
            @test A * B ≈ A * (a' .+ A)
            B = BroadcastArray(+, c, A)
            @test B * b ≈ (c .+ A) * b
            @test B * A ≈ (c .+ A) * A
            @test A * B ≈ A * (c .+ A)
            B = BroadcastArray(+, c, b)
            @test B * reshape(b,1,5) ≈ (c .+ b) * reshape(b,1,5)
        end

        @testset "-" begin
            B = BroadcastArray(-, a, A)
            @test B * b ≈ (a .- A) * b
            @test B * A ≈ (a .- A) * A
            @test A * B ≈ A * (a .- A)
            B = BroadcastArray(-, A, a)
            @test B * b ≈ (A .- a) * b
            @test B * A ≈ (A .- a) * A
            @test A * B ≈ A * (A .- a)
            B = BroadcastArray(-, A, 2A)
            @test B * b ≈ (A .- 2A) * b
            @test B * A ≈ (A .- 2A) * A
            @test A * B ≈ A * (A .- 2A)
            B = BroadcastArray(-, ã, A)
            @test B * b ≈ (ã .- A) * b
            @test B * A ≈ (ã .- A) * A
            B = BroadcastArray(-, A, ã)
            @test B * b ≈ (A .- ã) * b
            @test B * A ≈ (A .- ã) * A
            B = BroadcastArray(-, a', A)
            @test B * b ≈ (a' .- A) * b
            @test B * A ≈ (a' .- A) * A
            B = BroadcastArray(-, c, A)
            @test B * b ≈ (c .- A) * b
            @test B * A ≈ (c .- A) * A
        end

        @testset "Mixed" begin
            B = BroadcastArray(+, A, 2A)
            C = BroadcastArray(-, A, 2A)
            @test B*C ≈ 3A * (-A)
            @test C*B ≈ (-A) * 3A 
        end
    end
end