using LazyArrays, LinearAlgebra,  Test
import LazyArrays: ArrayLdivArrayStyle
import Base.Broadcast: materialize

@testset "Ldiv" begin
    @testset "Float64 \\ *" begin
        A = randn(5,5)
        b = randn(5)
        M = Mul(Inv(A),b)

        @test size(M) == (5,)
        @test similar(M) isa Vector{Float64}
        @test materialize(M) isa Vector{Float64}
        @test all(materialize(M) .=== (A\b))

        @test Base.BroadcastStyle(typeof(Ldiv(A,b))) isa ArrayLdivArrayStyle
        @test all(copyto!(similar(b), Ldiv(A,b)) .===
                    (similar(b) .= Ldiv(A,b)) .=== Inv(A) * b .===
                    materialize(Ldiv(A,b)) .===
                  (A\b) .=== (b̃ =  copy(b); LAPACK.gesv!(copy(A), b̃); b̃))


        @test copyto!(similar(b), Ldiv(UpperTriangular(A) , b)) ≈ UpperTriangular(A) \ b
        @test all(copyto!(similar(b), Ldiv(UpperTriangular(A) , b)) .===
                    (similar(b) .= Ldiv(UpperTriangular(A),b)) .=== Inv(UpperTriangular(A))*b .===
                    BLAS.trsv('U', 'N', 'N', A, b) )


        @test copyto!(similar(b), Ldiv(UpperTriangular(A)' , b)) ≈ UpperTriangular(A)' \ b
        @test all(copyto!(similar(b), Ldiv(UpperTriangular(A)' , b)) .===
                    (similar(b) .= Ldiv(UpperTriangular(A)',b)) .===
                    copyto!(similar(b), Ldiv(transpose(UpperTriangular(A)) , b)) .===
                            (similar(b) .= Ldiv(transpose(UpperTriangular(A)),b)) .===
                    BLAS.trsv('U', 'T', 'N', A, b))

        b = randn(5) + im*randn(5)
        @test Inv(A) * b ≈ Matrix(A) \ b
    end


    @testset "ComplexF64 \\ *" begin
        T = ComplexF64
        A = randn(T,5,5)
        b = randn(T,5)
        @test Base.BroadcastStyle(typeof(Ldiv(A,b))) isa ArrayLdivArrayStyle
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
        @test Inv(A) * b ≈ A \ b
    end

    @testset "Rectangle PInv" begin
        A = randn(5,3)
        A_orig = copy(A)
        b = randn(5)
        b_orig = copy(b)
        @test_throws DimensionMismatch Inv(A)
        @test PInv(A) * b == (A\b)
        @test all(b .=== b_orig)
    end
end
