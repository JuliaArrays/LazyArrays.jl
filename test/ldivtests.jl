using LazyArrays, LinearAlgebra,  Test
import LazyArrays: ArrayLdivArrayStyle
@testset "Ldiv" begin
    A = randn(5,5)
    b = randn(5)
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
                copyto!(similar(b), Ldiv(transpose(UpperTriangular(A)) , b)) .===
                        (similar(b) .= Ldiv(transpose(UpperTriangular(A)),b)) .===
                BLAS.trsv('U', 'T', 'N', A, b))

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
end
