module AddTests

using LazyArrays, Test
using LinearAlgebra
import LazyArrays: Add, AddArray, MulAdd, materialize!, MemoryLayout, ApplyLayout, simplifiable, simplify

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

        for op in (+, -)
            @testset "$op" begin
                @testset "vec .$op mat" begin
                    B = BroadcastArray(op, a, A)
                    @test @inferred(B * b) isa Vector
                    @test @inferred(B * A) isa Matrix
                    @test @inferred(A * B) isa Matrix
                    @test B * B isa Matrix
                    @test B * b ≈ op.(a, A) * b
                    @test B * A ≈ op.(a, A) * A
                    @test A * B ≈ A * op.(a, A)
                    @test B * B ≈ op.(a, A) * op.(a, A)
                end
                @testset "mat .$op vec" begin
                    B = BroadcastArray(op, A, a)
                    @test B * b ≈ op.(A, a) * b
                    @test B * A ≈ op.(A, a) * A
                    @test A * B ≈ A * op.(A, a)
                end
                @testset "mat .$op mat" begin
                    B = BroadcastArray(op, A, 2A)
                    @test B * b ≈ op.(A, 2A) * b
                    @test B * A ≈ op.(A, 2A) * A
                    @test A * B ≈ A * op.(A, 2A)
                end
                @testset "vecmat .$op mat" begin
                    B = BroadcastArray(op, ã, A)
                    @test B * b ≈ op.(ã, A) * b
                    @test B * A ≈ op.(ã, A) * A
                    @test A * B ≈ A * op.(ã, A)
                end
                @testset "mat .$op vecmat" begin
                    B = BroadcastArray(op, A, ã)
                    @test B * b ≈ op.(A, ã) * b
                    @test B * A ≈ op.(A, ã) * A
                    @test A * B ≈ A * op.(A, ã)
                end
                @testset "rowvec .$op mat" begin
                    B = BroadcastArray(op, a', A)
                    @test B * b ≈ op.(a', A) * b
                    @test B * A ≈ op.(a', A) * A
                    @test A * B ≈ A * op.(a', A)
                end
                @testset "constvec .$op mat" begin
                    B = BroadcastArray(op, c, A)
                    B̃ = BroadcastArray(op, c[1], A)
                    @test B * b ≈ B̃ * b ≈ op.(c, A) * b
                    @test B * A ≈ B̃ * A ≈ op.(c, A) * A
                    @test A * B ≈ A * B̃ ≈ A * op.(c, A)
                end
                @testset "constvec .$op vec" begin
                    B = BroadcastArray(op, c, b)
                    @test B * reshape(b,1,5) ≈ op.(c, b) * reshape(b,1,5)
                end
            end
        end

        @testset "Mixed" begin
            B = BroadcastArray(+, A, 2A)
            C = BroadcastArray(-, A, 2A)
            @test B*C ≈ 3A * (-A)
            @test C*B ≈ (-A) * 3A
        end

        @testset "simplifiable" begin
            A = randn(5,5)
            B = BroadcastArray(+, A, 2A)
            C = BroadcastArray(-, A, 2A)
            D = ApplyArray(exp, A)
            @test simplifiable(*, A, B) == Val(true)
            @test simplifiable(*, B, A) == Val(true)
            @test simplifiable(*, A, C) == Val(true)
            @test simplifiable(*, C, A) == Val(true)
            @test simplifiable(*, B, C) == Val(true)
            @test simplifiable(*, C, B) == Val(true)
            @test simplifiable(*, B, B) == Val(true)
            @test simplifiable(*, C, C) == Val(true)
            @test simplifiable(*, B, D) == Val(true)
            @test simplifiable(*, D, B) == Val(true)

            @test A*B ≈ simplify(Mul(A,B)) ≈ A * Matrix(B)
            @test B*A ≈ simplify(Mul(B,A)) ≈ Matrix(B)A
            @test A*C ≈ simplify(Mul(A,C)) ≈ A * Matrix(C)
            @test C*A ≈ simplify(Mul(C,A)) ≈ Matrix(C)A
            @test B*C ≈ simplify(Mul(B,C)) ≈ B * Matrix(C)
            @test C*B ≈ simplify(Mul(C,B)) ≈ Matrix(C)B
            @test B*B ≈ simplify(Mul(B,B)) ≈ Matrix(B)B
            @test C*C ≈ simplify(Mul(C,C)) ≈ Matrix(C)C
            @test B*D ≈ simplify(Mul(B,D)) ≈ Matrix(B)*D
            @test D*B ≈ simplify(Mul(D,B)) ≈ D*Matrix(B)
        end

        @testset "Add * Mul" begin
            A = randn(5,5)
            B = BroadcastArray(+, A, 2A)
            C = BroadcastArray(-, A, 2A)
            M = ApplyArray(*, A, A)
            @test B*M ≈ Matrix(B)*M
            @test C*M ≈ Matrix(C)*M
            @test M*B ≈ M*Matrix(B)
            @test M*C ≈ M*Matrix(C)
            @test simplifiable(*,B,M) == simplifiable(*,C,M) == Val(true)
            @test simplifiable(*,M,B) == simplifiable(*,M,C) == Val(true)
        end
    end
end # testset

end # module
