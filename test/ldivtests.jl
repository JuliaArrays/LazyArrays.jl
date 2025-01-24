module LdivRdivTests

using LazyArrays, LinearAlgebra, FillArrays, Test, ArrayLayouts 
import ArrayLayouts: rowsupport, colsupport
import LazyArrays: InvMatrix, ApplyBroadcastStyle, LdivStyle, Applied, LazyLayout, simplifiable
import Base.Broadcast: materialize

@testset "Ldiv" begin
    @testset "Float64 \\ *" begin
        A = randn(5,5)
        b = randn(5)
        M = Ldiv(A,b)
        @test all(materialize(M) .=== (A\b) .=== materialize(applied(\,A,b)))

        @test applied(\,A,b) isa Applied{LdivStyle}
        @test applied(\,A,b)[1] ≈ (A\b)[1]

        @test parent(InvMatrix(A)) === inv(InvMatrix(A)) === pinv(InvMatrix(A)) === A
        @test parent(PInvMatrix(A)) === pinv(InvMatrix(A)) === A

        @test InvMatrix(A)[1,2] ≈ PInvMatrix(A)[1,2] ≈ inv(A)[1,2]

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
        @test InvMatrix(A) \ b == A*b
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

        A = UpperTriangular(Ones(20,20))
        @test ApplyArray(inv,A)[1:10,1:10] ≈ diagm(0 => ones(10), 1 => -ones(9))
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
        M = ApplyArray(*, A, B)
        b = 1:5
        @test apply(\,B,ApplyArray(*,A,b)) ≈ B\A*b
        @test M \ b ≈ B\(A\b)
        @test M \ M ≈ Eye(5)
        @test M \ b isa Vector
        @test M\M isa Matrix

        @testset "Inv of Mul" begin
            @test inv(M) ≈ inv(A*B)
            @test_throws DimensionMismatch inv(ApplyArray(*,randn(5,6), rand(6,5)))
        end

        @testset "Inv * Eye" begin
            @test ApplyArray(inv,A) * Eye(5) == Eye(5) * ApplyArray(inv,A)
            @test ApplyArray(inv,A) * Diagonal(Fill(2,5)) == Diagonal(Fill(2,5)) * ApplyArray(inv,A)
        end
    end

    @testset "Triangular ldiv" begin
        A = randn(5,5)
        b = randn(5)
        L = applied(\, UpperTriangular(A), b)
        @test copyto!(similar(L), L) ≈ UpperTriangular(A) \ b ≈ materialize!(L)
    end

    @testset "Diagonal \\ Mul" begin
        D = Diagonal(randn(5))
        A = ApplyArray(*, randn(5,5), randn(5))
        @test D \ A ≈ D \ Vector(A)
    end

    @testset "LdivArray Mul" begin
        A = ApplyArray(\, Diagonal(1:5), randn(5,5))
        b = randn(5)
        @test A * b ≈ Matrix(A) * b
    end

    @testset "Diagonal \\ Lazy" begin
        D = Diagonal(randn(5))
        A = BroadcastArray(*, randn(5,5), randn(5))
        @test D \ A ≈ D \ Matrix(A)
    end

    @testset "simplifiable" begin
        A = randn(5,5)
        b = randn(5)
        @test simplifiable(\, A, b) isa Val{true}
        B = ApplyArray(*, randn(5,5), b)
        @test simplifiable(\, A, B) isa Val{true}
    end
end

@testset "Rdiv" begin
    A = randn(5,5)
    R = ApplyArray(/, A, A)
    @test axes(R) == (axes(R,1),axes(R,2)) == axes(A)
    @test R ≈ apply(/, A, A) ≈ Eye(5)
end

@testset "UpperTriangular inv (#320)" begin
    A = randn(5,5)
    B = randn(5,5)
    M = ApplyArray(*, A, B)
    @test inv(UpperTriangular(M)) isa ApplyArray
    @test inv(UpperTriangular(M)) ≈ inv(UpperTriangular(Matrix(M)))
end

@testset "Issue #316" begin
    X = [1.0 2.0; 3.0 4.0]
    Y = ApplyArray(inv, ApplyArray(*, X, [1.0 0.0; 0.0 1.0]))
    Z = Diagonal(Ones(2))
    @test Z \ Y ≈ [-2.0 1.0; 1.5 -0.5]
end

@testset "Issue #329" begin
    U = UpperTriangular(ApplyArray(inv, rand(5, 5)))
    invU = inv(U)
    L = LowerTriangular(ApplyArray(inv, rand(10, 10)))
    invL = inv(L)
    @test colsupport(invU, 1) == 1:1 
    @test colsupport(invU, 3) == 1:3
    @test colsupport(invU, 1:3) == 1:3 
    @test rowsupport(invU, 1) == 1:5 
    @test rowsupport(invU, 4) == 4:5 
    @test rowsupport(invU, 5) == 5:5 
    @test rowsupport(invU, 3:5) == 3:5
    @test colsupport(invL, 1) == 1:10 
    @test colsupport(invL, 5) == 5:10 
    @test colsupport(invL, 2:4) == 2:10
    @test rowsupport(invL, 1) == 1:1 
    @test rowsupport(invL, 4) == 1:4 
    @test rowsupport(invL, 2:2:4) == 1:4
    @test colsupport(invU, ()) == 1:0
    @test rowsupport(invU, ()) == 1:0 
    @test colsupport(invL, ()) == 1:0 
    @test rowsupport(invL, ()) == 1:0
end

@testset "Inv \\ Lazy" begin
    A = randn(5,5)
    Ai = InvMatrix(A)
    @test Ai \ Ai ≈ I
    @test Ai \ BroadcastArray(exp, A) ≈ Ai \ exp.(A) ≈ A*exp.(A)
    @test Ai \ ApplyArray(*, A, A) ≈ A^3
end

end # module