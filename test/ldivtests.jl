using LazyArrays, LinearAlgebra, Test
import LazyArrays: InvMatrix, ApplyBroadcastStyle, LdivApplyStyle, Applied
import Base.Broadcast: materialize

@testset "Ldiv" begin
    @testset "Float64 \\ *" begin
        A = randn(5,5)
        b = randn(5)
        M = Ldiv(A,b)

        @test size(M) == (5,)
        @test similar(M) isa Vector{Float64}
        @test materialize(M) isa Vector{Float64}
        @test all(materialize(M) .=== (A\b) .=== materialize(applied(\,A,b)))

        @test Base.BroadcastStyle(typeof(Ldiv(A,b))) isa ApplyBroadcastStyle
        @test applied(\,A,b) isa Applied{LdivApplyStyle}

        @test all(copyto!(similar(b), Ldiv(A,b)) .===
                    (similar(b) .= Ldiv(A,b)) .=== InvMatrix(A) * b .===
                    materialize(Ldiv(A,b)) .===
                    apply(\,A,b) .===
                  (A\b) .=== (b̃ =  copy(b); LAPACK.gesv!(copy(A), b̃); b̃))

        @test copyto!(similar(b), Ldiv(UpperTriangular(A) , b)) ≈ UpperTriangular(A) \ b
        @test all(copyto!(similar(b), Ldiv(UpperTriangular(A) , b)) .===
                    (similar(b) .= Ldiv(UpperTriangular(A),b)) .===
                    InvMatrix(UpperTriangular(A))*b .===
                    BLAS.trsv('U', 'N', 'N', A, b) )

        @test copyto!(similar(b), Ldiv(UpperTriangular(A)' , b)) ≈ UpperTriangular(A)' \ b
        @test all(copyto!(similar(b), Ldiv(UpperTriangular(A)' , b)) .===
                    (similar(b) .= Ldiv(UpperTriangular(A)',b)) .===
                    copyto!(similar(b), Ldiv(transpose(UpperTriangular(A)) , b)) .===
                            (similar(b) .= Ldiv(transpose(UpperTriangular(A)),b)) .===
                    BLAS.trsv('U', 'T', 'N', A, b))

        b = randn(5) + im*randn(5)
        @test InvMatrix(A) * b ≈ Matrix(A) \ b
    end

    @testset "ComplexF64 \\ *" begin
        T = ComplexF64
        A = randn(T,5,5)
        b = randn(T,5)
        @test all(copyto!(similar(b), Ldiv(A,b)) .===
                    (similar(b) .= Ldiv(A,b)) .===
                  (A\b) .=== (b̃ =  copy(b); LAPACK.gesv!(copy(A), b̃); b̃))

        @test copyto!(similar(b), Ldiv(UpperTriangular(A) , b)) ≈ UpperTriangular(A) \ b
        @test all(copyto!(similar(b), Ldiv(UpperTriangular(A) , b)) .===
                    (similar(b) .= Ldiv(UpperTriangular(A),b)) .===
                    BLAS.trsv('U', 'N', 'N', A, b) )

        @test copyto!(similar(b), Ldiv(UpperTriangular(A)' , b)) ≈ UpperTriangular(A)' \ b
        @test all(copyto!(similar(b), Ldiv(UpperTriangular(A)' , b)) .===
                    (similar(b) .= Ldiv(UpperTriangular(A)',b)) .===
                    BLAS.trsv('U', 'C', 'N', A, b))

        @test copyto!(similar(b), Ldiv(transpose(UpperTriangular(A)) , b)) ≈ transpose(UpperTriangular(A)) \ b
        @test all(copyto!(similar(b), Ldiv(transpose(UpperTriangular(A)) , b)) .===
                            (similar(b) .= Ldiv(transpose(UpperTriangular(A)),b)) .===
                    BLAS.trsv('U', 'T', 'N', A, b))

        b = randn(5)
        @test InvMatrix(A) * b ≈ A \ b
    end

    @testset "Triangular \\ matrix" begin
        A = randn(5,5)
        b = randn(5,5)
        M =  Ldiv(UpperTriangular(A), b)
        @test Base.Broadcast.broadcastable(M) === M
        @test UpperTriangular(A) \ b ≈ copyto!(similar(b) , Ldiv(UpperTriangular(A), b)) ≈
            (b .= Ldiv(UpperTriangular(A), b))
    end

    @testset "Rectangle PInv" begin
        A = randn(5,3)
        A_orig = copy(A)
        b = randn(5)
        b_orig = copy(b)
        @test_throws DimensionMismatch InvMatrix(A)
        @test PInvMatrix(A) * b == (A\b)
        @test all(b .=== b_orig)
    end

    @testset "inv and pinv"  begin
        A = randn(5,5)
        @test inv(Inv(A)) === inv(PInv(A)) === pinv(Inv(A)) === pinv(PInv(A)) === A

        Ai = PInvMatrix(A)
        @test Matrix(Ai) ≈ inv(A)
        Ai = InvMatrix(A)
        @test Matrix(Ai) ≈ inv(A)


        A = randn(5,3)
        @test pinv(PInv(A)) === A
        Ai = PInvMatrix(A)
        @test Ai ≈ pinv(A)
        @test_throws DimensionMismatch inv(PInv(A))
        @test_throws DimensionMismatch InvMatrix(A)

        A = randn(3,5)
        @test pinv(PInv(A)) === A
        Ai = PInvMatrix(A)
        @test Ai ≈ pinv(A)
        @test_throws DimensionMismatch inv(PInv(A))
        @test_throws DimensionMismatch InvMatrix(A)
    end
end
