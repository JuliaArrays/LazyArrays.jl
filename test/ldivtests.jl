using LazyArrays, LinearAlgebra, Test
import LazyArrays: InvMatrix, ApplyBroadcastStyle, LdivStyle, Applied, LazyLayout
import Base.Broadcast: materialize

@testset "Ldiv" begin
    @testset "Float64 \\ *" begin
        A = randn(5,5)
        b = randn(5)
        M = Ldiv(A,b)
        @test all(materialize(M) .=== (A\b) .=== materialize(applied(\,A,b)))

        @test applied(\,A,b) isa Applied{LdivStyle}

        @test parent(InvMatrix(A)) === A

        @test all(copyto!(similar(b), Ldiv(A,b)) .===
                    (similar(b) .= Ldiv(A,b)) .=== InvMatrix(A) * b .===
                    materialize(Ldiv(A,b)) .===
                    apply(\,A,b) .===
                  (A\b) .=== (b̃ =  copy(b); LAPACK.gesv!(copy(A), b̃); b̃))

        @test all(copyto!(similar(b), Ldiv(UpperTriangular(A) , b)) .===
                    (similar(b) .= Ldiv(UpperTriangular(A),b)) .===
                    InvMatrix(UpperTriangular(A))*b .===
                    BLAS.trsv('U', 'N', 'N', A, b) )

        b = randn(5) + im*randn(5)
        @test InvMatrix(A) * b ≈ Matrix(A) \ b
    end

    @testset "ComplexF64 \\ *" begin
        T = ComplexF64
        A = randn(T,5,5)
        b = randn(T,5)

        b = randn(5)
        @test InvMatrix(A) * b ≈ A \ b
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

    @testset "Int" begin
        A = [1 2 ; 3 4]; b = [5,6];
        @test eltype(applied(inv, A)) == eltype(applied(pinv, A)) == eltype(applied(\, A, b)) == eltype(Ldiv(A, b)) == 
            eltype(ApplyArray(\, A, b)) == eltype(InvMatrix(A)) == eltype(PInvMatrix(A)) == Float64
        @test apply(\,A,b) == A\b
    end

    @testset "QR" begin
        A = randn(5,3)
        b = randn(5)
        B = randn(5,5)
        Q,R = qr(A)
        @test Q\b ≈ apply(\,Q,b) 
        @test Q\B ≈ apply(\,Q,B)
        @test_throws DimensionMismatch apply(\, Q, randn(4))
        @test_throws DimensionMismatch apply(\, Q, randn(4,3))
        dest = fill(NaN,5)
        @test copyto!(dest, applied(\,Q,b)) == apply(\,Q,b)
    end

    @testset "Lazy *" begin
        A = randn(5,5)
        B = randn(5,5)
        b = 1:5
        @test apply(\,B,ApplyArray(*,A,b)) == B\A*b
    end
end
