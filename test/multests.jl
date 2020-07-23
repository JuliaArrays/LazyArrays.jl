using Test, LinearAlgebra, LazyArrays, StaticArrays, FillArrays
import LazyArrays: MulAdd, MemoryLayout, DenseColumnMajor, DiagonalLayout, SymTridiagonalLayout, Add, AddArray, 
                    MulAddStyle, Applied, ApplyStyle, LmulStyle, Lmul, ApplyArrayBroadcastStyle, DefaultArrayApplyStyle,
                    FlattenMulStyle, RmulStyle, Rmul, ApplyLayout, arguments, colsupport, rowsupport
import Base.Broadcast: materialize, materialize!, broadcasted
import MatrixFactorizations: QRCompactWYQLayout, AdjQRCompactWYQLayout

@testset "Matrix * Vector" begin
    @testset "eltype" begin
        @test @inferred(eltype(Mul(zeros(Int,2,2), zeros(Float64,2)))) == Float64
        @test @inferred(eltype(Mul(zeros(ComplexF16,2,2),zeros(Int,2,2),zeros(Float64,2)))) == ComplexF64

        v = Mul(zeros(Int,2,2), zeros(Float64,2))
        A = Mul(zeros(Int,2,2), zeros(Float64,2,2))
        @test @inferred(axes(v)) == (@inferred(axes(v,1)),) == (Base.OneTo(2),)
        @test @inferred(size(v)) == (@inferred(size(v,1)),) == (2,)
        @test @inferred(axes(A)) == (@inferred(axes(A,1)),@inferred(axes(A,2))) == (Base.OneTo(2),Base.OneTo(2))
        @test @inferred(size(A)) == (@inferred(size(A,1)),size(A,2)) == (2,2)

        Ã = materialize(A)
        @test Ã isa Matrix{Float64}
        @test Ã == zeros(2,2)

        c = similar(v)
        fill!(c,NaN)
        @test copyto!(c,v) == zeros(2)
        fill!(c,NaN)
        c .= v
        @test c == zeros(2)

        A = randn(6,5); B = randn(5,5); C = randn(5,6)
        M = Mul(A,B,C)
        @test @inferred(eltype(M)) == Float64
        @test materialize(M) == A*B*C
    end

    @testset "gemv Float64" begin
        for A in (randn(5,5), view(randn(5,5),:,:), view(randn(5,5),1:5,:),
                  view(randn(5,5),1:5,1:5), view(randn(5,5),:,1:5)),
            b in (randn(5), view(randn(5),:), view(randn(5),1:5), view(randn(9),1:2:9))
            c = similar(b);

            c .= applied(*,A,b)
            @test applied(*,A,b) isa Mul{MulAddStyle}
            @test all(c .=== A*b .=== BLAS.gemv!('N', 1.0, A, b, 0.0, similar(c)))
            copyto!(c, applied(*,A,b))
            @test all(c .=== A*b .=== BLAS.gemv!('N', 1.0, A, b, 0.0, similar(c)))
            c .= @~ A*b
            @test all(c .=== A*b .=== BLAS.gemv!('N', 1.0, A, b, 0.0, similar(c)))

            b̃ = copy(b)
            copyto!(b̃, applied(*,A,b̃))
            @test all(c .=== b̃)

            M = applied(*, 2.0, A,b)
            @test M isa Applied{MulAddStyle}
            c .= M
            @test all(c .=== BLAS.gemv!('N', 2.0, A, b, 0.0, similar(c)))
            copyto!(c, M)
            @test all(c .=== BLAS.gemv!('N', 2.0, A, b, 0.0, similar(c)))
            c .= @~ 2.0*A*b
            @test all(c .=== BLAS.gemv!('N', 2.0, A, b, 0.0, similar(c)))


            c = copy(b)
            M = applied(+, applied(*,A,b), c)
            @test M isa Applied{MulAddStyle}
            copyto!(c, M)
            @test all(c .=== BLAS.gemv!('N', 1.0, A, b, 1.0, copy(b)))
            c = copy(b)
            c .= applied(+, applied(*,A,b), c)
            @test all(c .=== BLAS.gemv!('N', 1.0, A, b, 1.0, copy(b)))
            c = copy(b)
            c .= @~ A*b + c
            @test all(c .=== BLAS.gemv!('N', 1.0, A, b, 1.0, copy(b)))

            c = copy(b)
            @test (@~ A*b + 2.0*c) isa Applied{MulAddStyle}
            c .= @~ A*b + 2.0*c
            @test all(c .=== BLAS.gemv!('N', 1.0, A, b, 2.0, copy(b)))

            c = copy(b)
            @test (@~ 2.0*A*b + c) isa Applied{MulAddStyle}
            c .= @~ 2.0*A*b + c
            @test all(c .=== BLAS.gemv!('N', 2.0, A, b, 1.0, copy(b)))

            c = copy(b)
            @test ( @~ 3.0*A*b + 2.0*c) isa Applied{MulAddStyle}
            c .= @~ 3.0*A*b + 2.0*c
            @test all(c .=== BLAS.gemv!('N', 3.0, A, b, 2.0, copy(b)))

            d = similar(c)
            c = copy(b)
            d .= @~ 3.0*A*b + 2.0*c
            @test all(d .=== BLAS.gemv!('N', 3.0, A, b, 2.0, copy(b)))
        end
    end

    @testset "gemv mixed array types" begin
        (A, b, c) = (randn(5,5), randn(5), 1.0:5.0)
        d = similar(b)
        d .= @~ A*b + c
        @test all(d .=== BLAS.gemv!('N', 1.0, A, b, 1.0, Vector{Float64}(c)))

        d .= @~ A*b + 2.0*c
        @test all(d .=== BLAS.gemv!('N', 1.0, A, b, 2.0, Vector{Float64}(c)))

        d .= @~ 2.0*A*b + c
        @test all(d .=== BLAS.gemv!('N', 2.0, A, b, 1.0, Vector{Float64}(c)))

        d .= @~ 3.0*A*b + 2.0 * c
        @test all(d .=== BLAS.gemv!('N', 3.0, A, b, 2.0, Vector{Float64}(c)))

        @test (similar(b) .= @~ 1 * A * b) ≈
            (similar(b) .= @~ 1 * A *b + 0 * b) ≈
            A*b
    end

    @testset "gemv Complex" begin
        for T in (ComplexF64,),
                A in (randn(T,5,5), view(randn(T,5,5),:,:)),
                b in (randn(T,5), view(randn(T,5),:))
            c = similar(b);

            c .= @~ A*b
            @test all(c .=== BLAS.gemv!('N', one(T), A, b, zero(T), similar(c)))

            b .= @~ A*b
            @test all(c .=== b)

            c .= applied(*, 2one(T), A, b)
            @test all(c .=== BLAS.gemv!('N', 2one(T), A, b, zero(T), similar(c)))

            c = copy(b)
            c .= @~ A*b + c
            @test all(c .=== BLAS.gemv!('N', one(T), A, b, one(T), copy(b)))


            c = copy(b)
            c .= applied(+, @~(A*b), applied(*, 2one(T), c))
            @test all(c .=== BLAS.gemv!('N', one(T), A, b, 2one(T), copy(b)))

            c = copy(b)
            c .= applied(+, applied(*, 2one(T), A, b), c)
            @test all(c .=== BLAS.gemv!('N', 2one(T), A, b, one(T), copy(b)))


            c = copy(b)
            c .= applied(+, applied(*, 3one(T), A, b), applied(*, 2one(T), c))
            @test all(c .=== BLAS.gemv!('N', 3one(T), A, b, 2one(T), copy(b)))

            d = similar(c)
            c = copy(b)
            d .= applied(+, applied(*, 3one(T), A, b), applied(*, 2one(T), c))
            @test all(d .=== BLAS.gemv!('N', 3one(T), A, b, 2one(T), copy(b)))
        end
    end
end
@testset "Matrix * Matrix" begin
    @testset "gemm" begin
        for A in (randn(5,5), view(randn(5,5),:,:), view(randn(5,5),1:5,:),
                  view(randn(5,5),1:5,1:5), view(randn(5,5),:,1:5)),
            B in (randn(5,5), view(randn(5,5),:,:), view(randn(5,5),1:5,:),
                      view(randn(5,5),1:5,1:5), view(randn(5,5),:,1:5))
            C = similar(B);

            C .= @~ A*B
            @test all(C .=== BLAS.gemm!('N', 'N', 1.0, A, B, 0.0, similar(C)))

            B .= @~ A*B
            @test all(C .=== B)

            C .= @~ 2.0 * A*B
            @test all(C .=== BLAS.gemm!('N', 'N', 2.0, A, B, 0.0, similar(C)))

            C = copy(B)
            C .= @~ A*B + C
            @test all(C .=== BLAS.gemm!('N', 'N', 1.0, A, B, 1.0, copy(B)))


            C = copy(B)
            C .= @~ A*B + 2.0 * C
            @test all(C .=== BLAS.gemm!('N', 'N', 1.0, A, B, 2.0, copy(B)))

            C = copy(B)
            C .= @~ 2.0 * A*B + C
            @test all(C .=== BLAS.gemm!('N', 'N', 2.0, A, B, 1.0, copy(B)))


            C = copy(B)
            C .= @~ 3.0 * A*B + 2.0 * C
            @test all(C .=== BLAS.gemm!('N', 'N', 3.0, A, B, 2.0, copy(B)))

            d = similar(C)
            C = copy(B)
            d .= @~ 3.0 * A*B + 2.0 * C
            @test all(d .=== BLAS.gemm!('N', 'N', 3.0, A, B, 2.0, copy(B)))
        end
    end

    @testset "gemm Complex" begin
        for T in (ComplexF64,),
                A in (randn(T,5,5), view(randn(T,5,5),:,:)),
                B in (randn(T,5,5), view(randn(T,5,5),:,:))
            C = similar(B);

            C .= @~ A*B
            @test all(C .=== BLAS.gemm!('N', 'N', one(T), A, B, zero(T), similar(C)))

            B .= @~ A*B
            @test all(C .=== B)

            C .= applied(*, 2one(T), A, B)
            @test all(C .=== BLAS.gemm!('N', 'N', 2one(T), A, B, zero(T), similar(C)))

            C = copy(B)
            C .= @~ A*B + C
            @test all(C .=== BLAS.gemm!('N', 'N', one(T), A, B, one(T), copy(B)))


            C = copy(B)
            C .= applied(+, @~(A*B), applied(*, 2one(T), C))
            @test all(C .=== BLAS.gemm!('N', 'N', one(T), A, B, 2one(T), copy(B)))

            C = copy(B)
            C .= applied(+, applied(*, 2one(T), A,B), C)
            @test all(C .=== BLAS.gemm!('N', 'N', 2one(T), A, B, one(T), copy(B)))


            C = copy(B)
            C .= applied(+, applied(*, 3one(T),A,B), applied(*, 2one(T), C))
            @test all(C .=== BLAS.gemm!('N', 'N', 3one(T), A, B, 2one(T), copy(B)))

            d = similar(C)
            C = copy(B)
            d .= applied(+, applied(*,3one(T),A,B), applied(*,2one(T), C))
            @test all(d .=== BLAS.gemm!('N', 'N', 3one(T), A, B, 2one(T), copy(B)))
        end
    end

    @testset "gemm mixed array types" begin
        (A, B, C) = (randn(5,5), randn(5,5), reshape(1.0:25.0,5,5))
        D = similar(B)
        D .= @~ A*B + C
        @test all(D .=== BLAS.gemm!('N', 'N', 1.0, A, B, 1.0, Matrix{Float64}(C)))

        D .= @~ A*B + 2.0 * C
        @test all(D .=== BLAS.gemm!('N', 'N', 1.0, A, B, 2.0, Matrix{Float64}(C)))

        D .= @~ 2.0 * A*B + C
        @test all(D .=== BLAS.gemm!('N', 'N', 2.0, A, B, 1.0, Matrix{Float64}(C)))

        D .= @~ 3.0 * A*B + 2.0 * C
        @test all(D .=== BLAS.gemm!('N', 'N', 3.0, A, B, 2.0, Matrix{Float64}(C)))

        @test (similar(B) .= 1 * A*B) ≈
            (similar(B) .= 1 * A*B + 0 * B) ≈
            A*B
    end
end

@testset "adjtrans" begin
    @testset "gemv adjtrans" begin
        for A in (randn(5,5), view(randn(5,5),:,:), view(randn(5,5),1:5,:),
                  view(randn(5,5),1:5,1:5), view(randn(5,5),:,1:5)),
                    b in (randn(5), view(randn(5),:), view(randn(5),1:5), view(randn(9),1:2:9)),
                    Ac in (transpose(A), A')
            c = similar(b);

            c .= @~ Ac*b
            @test all(c .=== BLAS.gemv!('T', 1.0, A, b, 0.0, similar(c)))

            b .= @~ Ac*b
            @test all(c .=== b)

            c .= @~ 2.0 * Ac*b
            @test all(c .=== BLAS.gemv!('T', 2.0, A, b, 0.0, similar(c)))

            c = copy(b)
            c .= @~ Ac*b + c
            @test all(c .=== BLAS.gemv!('T', 1.0, A, b, 1.0, copy(b)))


            c = copy(b)
            c .= @~ Ac*b + 2.0 * c
            @test all(c .=== BLAS.gemv!('T', 1.0, A, b, 2.0, copy(b)))

            c = copy(b)
            c .= @~ 2.0 * Ac*b + c
            @test all(c .=== BLAS.gemv!('T', 2.0, A, b, 1.0, copy(b)))


            c = copy(b)
            c .= @~ 3.0 * Ac*b + 2.0 * c
            @test all(c .=== BLAS.gemv!('T', 3.0, A, b, 2.0, copy(b)))

            d = similar(c)
            c = copy(b)
            d .= @~ 3.0 * Ac*b + 2.0 * c
            @test all(d .=== BLAS.gemv!('T', 3.0, A, b, 2.0, copy(b)))
        end
    end

    @testset "gemv adjtrans mixed types" begin
        (A, b, c) = (randn(5,5), randn(5), 1.0:5.0)
        for Ac in (transpose(A), A')
            d = similar(b)
            d .= @~ Ac*b + c
            @test all(d .=== BLAS.gemv!('T', 1.0, A, b, 1.0, Vector{Float64}(c)))

            d .= @~ Ac*b + 2.0 * c
            @test all(d .=== BLAS.gemv!('T', 1.0, A, b, 2.0, Vector{Float64}(c)))

            d .= @~ 2.0 * Ac*b + c
            @test all(d .=== BLAS.gemv!('T', 2.0, A, b, 1.0, Vector{Float64}(c)))

            d .= @~ 3.0 * Ac*b + 2.0 * c
            @test all(d .=== BLAS.gemv!('T', 3.0, A, b, 2.0, Vector{Float64}(c)))
        end
    end

    @testset "gemv adjtrans Complex" begin
        for T in (ComplexF64,),
                A in (randn(T,5,5), view(randn(T,5,5),:,:)),
                b in (randn(T,5), view(randn(T,5),:)),
                (Ac,trans) in ((transpose(A),'T'), (A','C'))
            c = similar(b);

            c .= @~ Ac*b
            @test all(c .=== BLAS.gemv!(trans, one(T), A, b, zero(T), similar(c)))

            b .= @~ Ac*b
            @test all(c .=== b)

            c .= applied(*,2one(T),Ac,b)
            @test all(c .=== BLAS.gemv!(trans, 2one(T), A, b, zero(T), similar(c)))

            c = copy(b)
            c .= @~ Ac*b + c
            @test all(c .=== BLAS.gemv!(trans, one(T), A, b, one(T), copy(b)))


            c = copy(b)
            c .= applied(+, applied(*,Ac,b), applied(*,2one(T), c))
            @test all(c .=== BLAS.gemv!(trans, one(T), A, b, 2one(T), copy(b)))

            c = copy(b)
            c .= applied(+,applied(*,2one(T),Ac,b), c)
            @test all(c .=== BLAS.gemv!(trans, 2one(T), A, b, one(T), copy(b)))


            c = copy(b)
            c .= applied(+,applied(*,3one(T),Ac,b), applied(*,2one(T), c))
            @test all(c .=== BLAS.gemv!(trans, 3one(T), A, b, 2one(T), copy(b)))

            d = similar(c)
            c = copy(b)
            d .= applied(+,applied(*,3one(T),Ac,b), applied(*,2one(T), c))
            @test all(d .=== BLAS.gemv!(trans, 3one(T), A, b, 2one(T), copy(b)))
        end
    end

    @testset "gemm adjtrans" begin
        for A in (randn(5,5), view(randn(5,5),:,:)),
            B in (randn(5,5), view(randn(5,5),1:5,:))
            for Ac in (transpose(A), A')
                C = similar(B)
                C .= @~ Ac*B
                @test all(C .=== BLAS.gemm!('T', 'N', 1.0, A, B, 0.0, similar(C)))

                B .= @~ Ac*B
                @test all(C .=== B)

                C .= @~ 2.0 * Ac*B
                @test all(C .=== BLAS.gemm!('T', 'N', 2.0, A, B, 0.0, similar(C)))

                C = copy(B)
                C .= @~ Ac*B + C
                @test all(C .=== BLAS.gemm!('T', 'N', 1.0, A, B, 1.0, copy(B)))


                C = copy(B)
                C .= @~ Ac*B + 2.0 * C
                @test all(C .=== BLAS.gemm!('T', 'N', 1.0, A, B, 2.0, copy(B)))

                C = copy(B)
                C .= @~ 2.0 * Ac*B + C
                @test all(C .=== BLAS.gemm!('T', 'N', 2.0, A, B, 1.0, copy(B)))


                C = copy(B)
                C .= @~ 3.0 * Ac*B + 2.0 * C
                @test all(C .=== BLAS.gemm!('T', 'N', 3.0, A, B, 2.0, copy(B)))

                d = similar(C)
                C = copy(B)
                d .= @~ 3.0 * Ac*B + 2.0 * C
                @test all(d .=== BLAS.gemm!('T', 'N', 3.0, A, B, 2.0, copy(B)))
            end
            for Bc in (transpose(B), B')
                C = similar(B)
                C .= @~ A*Bc
                @test all(C .=== BLAS.gemm!('N', 'T', 1.0, A, B, 0.0, similar(C)))

                B .= @~ A*Bc
                @test all(C .=== B)

                C .= @~ 2.0 * A*Bc
                @test all(C .=== BLAS.gemm!('N', 'T', 2.0, A, B, 0.0, similar(C)))

                C = copy(B)
                C .= @~ A*Bc + C
                @test all(C .=== BLAS.gemm!('N', 'T', 1.0, A, B, 1.0, copy(B)))


                C = copy(B)
                C .= @~ A*Bc + 2.0 * C
                @test all(C .=== BLAS.gemm!('N', 'T', 1.0, A, B, 2.0, copy(B)))

                C = copy(B)
                C .= @~ 2.0 * A*Bc + C
                @test all(C .=== BLAS.gemm!('N', 'T', 2.0, A, B, 1.0, copy(B)))


                C = copy(B)
                C .= @~ 3.0 * A*Bc + 2.0 * C
                @test all(C .=== BLAS.gemm!('N', 'T', 3.0, A, B, 2.0, copy(B)))

                d = similar(C)
                C = copy(B)
                d .= @~ 3.0 * A*Bc + 2.0 * C
                @test all(d .=== BLAS.gemm!('N', 'T', 3.0, A, B, 2.0, copy(B)))
            end
            for Ac in (transpose(A), A'), Bc in (transpose(B), B')
                C = similar(B)
                C .= @~ Ac*Bc
                @test all(C .=== BLAS.gemm!('T', 'T', 1.0, A, B, 0.0, similar(C)))

                B .= @~ Ac*Bc
                @test all(C .=== B)

                C .= @~ 2.0 * Ac*Bc
                @test all(C .=== BLAS.gemm!('T', 'T', 2.0, A, B, 0.0, similar(C)))

                C = copy(B)
                C .= @~ Ac*Bc + C
                @test all(C .=== BLAS.gemm!('T', 'T', 1.0, A, B, 1.0, copy(B)))


                C = copy(B)
                C .= @~ Ac*Bc + 2.0 * C
                @test all(C .=== BLAS.gemm!('T', 'T', 1.0, A, B, 2.0, copy(B)))

                C = copy(B)
                C .= @~ 2.0 * Ac*Bc + C
                @test all(C .=== BLAS.gemm!('T', 'T', 2.0, A, B, 1.0, copy(B)))


                C = copy(B)
                C .= @~ 3.0 * Ac*Bc + 2.0 * C
                @test all(C .=== BLAS.gemm!('T', 'T', 3.0, A, B, 2.0, copy(B)))

                d = similar(C)
                C = copy(B)
                d .= @~ 3.0 * Ac*Bc + 2.0 * C
                @test all(d .=== BLAS.gemm!('T', 'T', 3.0, A, B, 2.0, copy(B)))
            end
        end
    end

    @testset "symv adjtrans" begin
        A = randn(100,100)
        x = randn(100)

        @test all( (similar(x) .= applied(*,Symmetric(A),x)) .===
                    (similar(x) .= applied(*,Hermitian(A),x)) .===
                    (similar(x) .= applied(*,Symmetric(A)',x)) .===
                    (similar(x) .= applied(*,Symmetric(view(A,:,:)',:L),x)) .===
                    (similar(x) .= applied(+,applied(*,1.0,Symmetric(A),x), applied(*,0.0,similar(x)))) .===
                    (similar(x) .= applied(*,view(Symmetric(A),:,:),x)) .===
                    BLAS.symv!('U', 1.0, A, x, 0.0, similar(x)) )

        y = copy(x)
        y .= applied(*,Symmetric(A), y)
        @test all( (similar(x) .= Mul(Symmetric(A),x)) .=== y)

        @test all( (similar(x) .= Mul(Symmetric(A,:L),x)) .===
                    (similar(x) .= Mul(Hermitian(A,:L),x)) .===
                    (similar(x) .= Mul(Symmetric(A,:L)',x)) .===
                    (similar(x) .= Mul(Symmetric(view(A,:,:)',:U),x)) .===
                    (fill!(similar(x),NaN) .= applied(+,applied(*,1.0,Symmetric(A,:L),x), applied(*,0.0,similar(x)))) .===
                    (fill!(similar(x),NaN) .= Mul(view(Hermitian(A,:L),:,:),x)) .===
                    BLAS.symv!('L', 1.0, A, x, 0.0, similar(x)) )
        T = ComplexF64
        A = randn(T,100,100)
        x = randn(T,100)
        @test all( (similar(x) .= Mul(Symmetric(A),x)) .===
                    (similar(x) .= Mul(transpose(Symmetric(A)),x)) .===
                    (similar(x) .= Mul(Symmetric(transpose(view(A,:,:)),:L),x)) .===
                    (similar(x) .= applied(+,applied(*,one(T),Symmetric(A),x), applied(*,zero(T),similar(x)))) .===
                    (similar(x) .= Mul(view(Symmetric(A),:,:),x)) .===
                    BLAS.symv!('U', one(T), A, x, zero(T), similar(x)) )

        @test all( (similar(x) .= Mul(Symmetric(A,:L),x)) .===
                    (similar(x) .= Mul(transpose(Symmetric(A,:L)),x)) .===
                    (similar(x) .= Mul(Symmetric(transpose(view(A,:,:)),:U),x)) .===
                    (similar(x) .= applied(+,applied(*,one(T),Symmetric(A,:L),x), applied(*,zero(T),similar(x)))) .===
                    (similar(x) .= Mul(view(Symmetric(A,:L),:,:),x)) .===
                    BLAS.symv!('L', one(T), A, x, zero(T), similar(x)) )

        @test all( (similar(x) .= Mul(Hermitian(A),x)) .===
                    (similar(x) .= Mul(Hermitian(A)',x)) .===
                    (similar(x) .= applied(+,applied(*,one(T),Hermitian(A),x), applied(*,zero(T),similar(x)))) .===
                    (similar(x) .= Mul(view(Hermitian(A),:,:),x)) .===
                    BLAS.hemv!('U', one(T), A, x, zero(T), similar(x)) )

        @test all( (similar(x) .= Mul(Hermitian(A,:L),x)) .===
                    (similar(x) .= Mul(Hermitian(A,:L)',x)) .===
                    (similar(x) .= applied(+,applied(*,one(T),Hermitian(A,:L),x), applied(*,zero(T),similar(x)))) .===
                    (similar(x) .= Mul(view(Hermitian(A,:L),:,:),x)) .===
                    BLAS.hemv!('L', one(T), A, x, zero(T), similar(x)) )

        y = copy(x)
        y .= Mul(Hermitian(A), y)
        @test all( (similar(x) .= Mul(Hermitian(A),x)) .=== y)
    end
end

@testset "Mul" begin
    @testset "Mixed types" begin
        A = randn(5,6)
        b = rand(Int,6)
        c = Array{Float64}(undef, 5)
        c .= applied(*,A,b)

        @test_throws DimensionMismatch (similar(c,3) .= Mul(A,b))
        @test_throws DimensionMismatch (c .= Mul(A,similar(b,2)))

        d = similar(c)
        mul!(d, A, b)
        @test all(c .=== d)

        copyto!(d, MulAdd(1, A, b, 0.0, d))
        @test d == copyto!(similar(d), MulAdd(1, A, b, 0.0, d)) ≈ A*b
        @test copyto!(similar(d), MulAdd(1, A, b, 1.0, d)) ≈ A*b + d

        @test all((similar(d) .= MulAdd(1, A, b, 1.0, d)) .=== copyto!(similar(d), MulAdd(1, A, b, 1.0, d)))

        B = rand(Int,6,4)
        C = Array{Float64}(undef, 5, 4)
        C .= Mul(A,B)

        @test_throws DimensionMismatch materialize!(MulAdd(1,A,B,0,similar(C,4,4)))
        @test_throws DimensionMismatch (similar(C,4,4) .= Mul(A,B))
        @test_throws DimensionMismatch (C .= Mul(A,similar(B,2,2)))

        D = similar(C)
        mul!(D, A, B)
        @test all(C .=== D)

        A = randn(Float64,20,22)
        B = randn(ComplexF64,22,24)
        C = similar(B,20,24)
        @test all((C .= Mul(A,B)  ) .=== copyto!(similar(C), MulAdd(1.0, A, B, 0.0, C)))
        @test copyto!(similar(C), MulAdd(1.0, A, B, 0.0, C)) ≈ A*B
    end

    @testset "no allocation" begin
        function blasnoalloc(c, α, A, x, β, y)
            c .= @~ A*x
            c .= @~ α*A*x
            c .= @~ A*x + y
            c .= @~ α * A*x + y
            c .= @~ A*x + β * y
            c .= @~ α * A*x + β * y
        end
        
        A = randn(5,5); x = randn(5); y = randn(5); c = similar(y);
        
        if VERSION ≥ v"1.1"
            @inferred(MulAdd(@~ A*x + y))
            @test blasnoalloc(c, 2.0, A, x, 3.0, y) === c
            @test @allocated(blasnoalloc(c, 2.0, A, x, 3.0, y)) == 0
            Ac = A'
            blasnoalloc(c, 2.0, Ac, x, 3.0, y)
            @test @allocated(blasnoalloc(c, 2.0, Ac, x, 3.0, y)) == 0
            Aa = ApplyArray(+, A, Ac)
            blasnoalloc(c, 2.0, Aa, x, 3.0, y)
			@test_broken @allocated(blasnoalloc(c, 2.0, Aa, x, 3.0, y)) == 0
		end
    end

    @testset "multi-argument mul" begin
        A = randn(5,5)
        B = apply(*,A,A,A)
        @test B isa Matrix{Float64}
        @test all(B .=== (A*A)*A)
    end

    @testset "#14" begin
        A = ones(1,1) * 1e200
        B = ones(1,1) * 1e150
        C = ones(1,1) * 1e-300

        @test apply(*, A, applied(*,B,C)) == A*(B*C)
        @test apply(*, A , applied(*,B,C), C) == A * (B*C) * C
    end

    @testset "#15" begin
        N = 10
        A = randn(N,N); B = randn(N,N); C = randn(N,N); R1 = similar(A); R2 = similar(A)
        M = Mul(A, Mul(B, C))
        @test axes(M) == (Base.OneTo(N),Base.OneTo(N))
        @test ndims(M) == ndims(typeof(M)) == 2
        @test eltype(M) == Float64
        @test all(copyto!(R1, M) .=== A*(B*C) .=== (R2 .= M))
    end

    @testset "broadcasting" begin
        A = randn(5,5); B = randn(5,5); C = randn(5,5)
        C .= NaN
        C .= @~ 1.0 * A*B + 0.0 * C
        @test C == A*B
    end

    @testset "BigFloat" begin
        A = BigFloat.(randn(5,5))
        x = BigFloat.(randn(5))
        @test A*x == apply(*,A,x) == copyto!(similar(x), applied(*,A,x))
        @test_throws UndefRefError materialize!(MulAdd(1.0,A,x,0.0,similar(x)))
    end

    @testset "Scalar * Vector" begin
        A, x =  [1 2; 3 4] , [[1,2],[3,4]]
        @test apply(*,A,x) == A*x
    end 

    @testset "Complex broadcast" begin
        A = randn(5,5) .+ im*randn(5,5)
        x = randn(5) .+ im*randn(5)
        y = randn(5) .+ im*randn(5)
        z = similar(x)
        @test all((z .= @~ (2.0+0.0im)*A*x + (3.0+0.0im)*y) .=== BLAS.gemv!('N',2.0+0.0im,A,x,3.0+0.0im,y))
    end
end

@testset "Lmul/Rmul" begin
    @testset "tri Lmul" begin
        @testset "Float * Float vector" begin
            A = randn(Float64, 100, 100)
            x = randn(Float64, 100)

            L = Lmul(UpperTriangular(A),x)
            @test size(L) == (size(L,1),) == (100,)
            @test axes(L) == (axes(L,1),) == (Base.OneTo(100),)
            @test eltype(L) == Float64
            @test length(L) == 100

            @test similar(L) isa Vector{Float64}
            @test similar(L,Int) isa Vector{Int}
            
            @test applied(*, UpperTriangular(A), x) isa Applied{LmulStyle}
            @test similar(applied(*, UpperTriangular(A), x), Float64) isa Vector{Float64}

            @test ApplyStyle(*, typeof(UpperTriangular(A)), typeof(x)) isa LmulStyle

            @test all((y = copy(x); y .= Mul(UpperTriangular(A),y) ) .===
                        (similar(x) .= Mul(UpperTriangular(A),x)) .===
                        copy(Lmul(UpperTriangular(A),x)) .===
                        materialize!(Lmul(UpperTriangular(A),copy(x))) .===
                        copyto!(similar(x),Lmul(UpperTriangular(A),x)) .===
                        UpperTriangular(A)*x .===
                        BLAS.trmv!('U', 'N', 'N', A, copy(x)))
            @test all((y = copy(x); y .= Mul(UnitUpperTriangular(A),y) ) .===
                        (similar(x) .= Mul(UnitUpperTriangular(A),x)) .===
                        UnitUpperTriangular(A)*x .===
                        BLAS.trmv!('U', 'N', 'U', A, copy(x)))
            @test all((y = copy(x); y .= Mul(LowerTriangular(A),y) ) .===
                        (similar(x) .= Mul(LowerTriangular(A),x)) .===
                        LowerTriangular(A)*x .===
                        BLAS.trmv!('L', 'N', 'N', A, copy(x)))
            @test all((y = copy(x); y .= Mul(UnitLowerTriangular(A),y) ) .===
                        (similar(x) .= Mul(UnitLowerTriangular(A),x)) .===
                        UnitLowerTriangular(A)x .===
                        BLAS.trmv!('L', 'N', 'U', A, copy(x)))

            @test all((y = copy(x); y .= Mul(UpperTriangular(A)',y) ) .===
                        (similar(x) .= Mul(UpperTriangular(A)',x)) .===
                        (similar(x) .= Mul(LowerTriangular(A'),x)) .===
                        UpperTriangular(A)'x .===
                        BLAS.trmv!('U', 'T', 'N', A, copy(x)))
            @test all((y = copy(x); y .= Mul(UnitUpperTriangular(A)',y) ) .===
                        (similar(x) .= Mul(UnitUpperTriangular(A)',x)) .===
                        (similar(x) .= Mul(UnitLowerTriangular(A'),x)) .===
                        UnitUpperTriangular(A)'*x .===
                        BLAS.trmv!('U', 'T', 'U', A, copy(x)))
            @test all((y = copy(x); y .= Mul(LowerTriangular(A)',y) ) .===
                        (similar(x) .= Mul(LowerTriangular(A)',x)) .===
                        (similar(x) .= Mul(UpperTriangular(A'),x)) .===
                        LowerTriangular(A)'*x .===
                        BLAS.trmv!('L', 'T', 'N', A, copy(x)))
            @test all((y = copy(x); y .= Mul(UnitLowerTriangular(A)',y) ) .===
                        (similar(x) .= Mul(UnitLowerTriangular(A)',x)) .===
                        (similar(x) .= Mul(UnitUpperTriangular(A'),x)) .===
                        UnitLowerTriangular(A)'*x .===
                        BLAS.trmv!('L', 'T', 'U', A, copy(x)))
        end

        @testset "Float * Complex vector"  begin
            T = ComplexF64
            A = randn(T, 100, 100)
            x = randn(T, 100)

            @test all((y = copy(x); y .= Mul(UpperTriangular(A),y) ) .===
                        (similar(x) .= Mul(UpperTriangular(A),x)) .===
                        UpperTriangular(A)x .===
                        BLAS.trmv!('U', 'N', 'N', A, copy(x)))
            @test all((y = copy(x); y .= Mul(UnitUpperTriangular(A),y) ) .===
                        (similar(x) .= Mul(UnitUpperTriangular(A),x)) .===
                        UnitUpperTriangular(A)x .===
                        BLAS.trmv!('U', 'N', 'U', A, copy(x)))
            @test all((y = copy(x); y .= Mul(LowerTriangular(A),y) ) .===
                        (similar(x) .= Mul(LowerTriangular(A),x)) .===
                        LowerTriangular(A)x .===
                        BLAS.trmv!('L', 'N', 'N', A, copy(x)))
            @test all((y = copy(x); y .= Mul(UnitLowerTriangular(A),y) ) .===
                        (similar(x) .= Mul(UnitLowerTriangular(A),x)) .===
                        UnitLowerTriangular(A)x .===
                        BLAS.trmv!('L', 'N', 'U', A, copy(x)))
            LowerTriangular(A')  == UpperTriangular(A)'

            @test all((y = copy(x); y .= Mul(transpose(UpperTriangular(A)),y) ) .===
                        (similar(x) .= Mul(transpose(UpperTriangular(A)),x)) .===
                        (similar(x) .= Mul(LowerTriangular(transpose(A)),x)) .===
                        transpose(UpperTriangular(A))x .===
                        BLAS.trmv!('U', 'T', 'N', A, copy(x)))
            @test all((y = copy(x); y .= Mul(transpose(UnitUpperTriangular(A)),y) ) .===
                        (similar(x) .= Mul(transpose(UnitUpperTriangular(A)),x)) .===
                        (similar(x) .= Mul(UnitLowerTriangular(transpose(A)),x)) .===
                        transpose(UnitUpperTriangular(A))x .===
                        BLAS.trmv!('U', 'T', 'U', A, copy(x)))
            @test all((y = copy(x); y .= Mul(transpose(LowerTriangular(A)),y) ) .===
                        (similar(x) .= Mul(transpose(LowerTriangular(A)),x)) .===
                        (similar(x) .= Mul(UpperTriangular(transpose(A)),x)) .===
                        transpose(LowerTriangular(A))x .===
                        BLAS.trmv!('L', 'T', 'N', A, copy(x)))
            @test all((y = copy(x); y .= Mul(transpose(UnitLowerTriangular(A)),y) ) .===
                        (similar(x) .= Mul(transpose(UnitLowerTriangular(A)),x)) .===
                        (similar(x) .= Mul(UnitUpperTriangular(transpose(A)),x)) .===
                        transpose(UnitLowerTriangular(A))x .===
                        BLAS.trmv!('L', 'T', 'U', A, copy(x)))

            @test all((y = copy(x); y .= Mul(UpperTriangular(A)',y) ) .===
                        (similar(x) .= Mul(UpperTriangular(A)',x)) .===
                        (similar(x) .= Mul(LowerTriangular(A'),x)) .===
                        UpperTriangular(A)'x .===
                        BLAS.trmv!('U', 'C', 'N', A, copy(x)))
            @test all((y = copy(x); y .= Mul(UnitUpperTriangular(A)',y) ) .===
                        (similar(x) .= Mul(UnitUpperTriangular(A)',x)) .===
                        (similar(x) .= Mul(UnitLowerTriangular(A'),x)) .===
                        UnitUpperTriangular(A)'x .===
                        BLAS.trmv!('U', 'C', 'U', A, copy(x)))
            @test all((y = copy(x); y .= Mul(LowerTriangular(A)',y) ) .===
                        (similar(x) .= Mul(LowerTriangular(A)',x)) .===
                        (similar(x) .= Mul(UpperTriangular(A'),x)) .===
                        LowerTriangular(A)'x .===
                        BLAS.trmv!('L', 'C', 'N', A, copy(x)))
            @test all((y = copy(x); y .= Mul(UnitLowerTriangular(A)',y) ) .===
                        (similar(x) .= Mul(UnitLowerTriangular(A)',x)) .===
                        (similar(x) .= Mul(UnitUpperTriangular(A'),x)) .===
                        UnitLowerTriangular(A)'x .===
                        BLAS.trmv!('L', 'C', 'U', A, copy(x)))
        end

        @testset "Float * Float Matrix" begin
            A = randn(Float64, 100, 100)
            x = randn(Float64, 100, 100)

            @test UpperTriangular(A)*x ≈ (similar(x) .= Mul(UpperTriangular(A), x))
        end

        @testset "adjtrans" begin
            for T in (Float64, ComplexF64)
                A = randn(T,100,100)
                b = randn(T,100)

                @test all(apply(*, UpperTriangular(A)', b) .=== UpperTriangular(A)'b)
                @test all(apply(*, UnitUpperTriangular(A)', b) .=== UnitUpperTriangular(A)'b)
                @test all(apply(*, LowerTriangular(A)', b) .=== LowerTriangular(A)'b)
                @test all(apply(*, UnitLowerTriangular(A)', b) .=== UnitLowerTriangular(A)'b)

                @test all(apply(*, transpose(UpperTriangular(A)), b) .=== transpose(UpperTriangular(A))b)
                @test all(apply(*, transpose(UnitUpperTriangular(A)), b) .=== transpose(UnitUpperTriangular(A))b)
                @test all(apply(*, transpose(LowerTriangular(A)), b) .=== transpose(LowerTriangular(A))b)
                @test all(apply(*, transpose(UnitLowerTriangular(A)), b) .=== transpose(UnitLowerTriangular(A))b)

                B = randn(T,100,100)

                @test all(apply(*, UpperTriangular(A)', B) .=== UpperTriangular(A)'B)
                @test all(apply(*, UnitUpperTriangular(A)', B) .=== UnitUpperTriangular(A)'B)
                @test all(apply(*, LowerTriangular(A)', B) .=== LowerTriangular(A)'B)
                @test all(apply(*, UnitLowerTriangular(A)', B) .=== UnitLowerTriangular(A)'B)

                @test all(apply(*, transpose(UpperTriangular(A)), B) .=== transpose(UpperTriangular(A))B)
                @test all(apply(*, transpose(UnitUpperTriangular(A)), B) .=== transpose(UnitUpperTriangular(A))B)
                @test all(apply(*, transpose(LowerTriangular(A)), B) .=== transpose(LowerTriangular(A))B)
                @test all(apply(*, transpose(UnitLowerTriangular(A)), B) .=== transpose(UnitLowerTriangular(A))B)                
            end

            for T in (Float64, ComplexF64)
                A = big.(randn(T,100,100))
                b = big.(randn(T,100))

                @test all(apply(*, UpperTriangular(A)', b) ≈ UpperTriangular(A)'b)
                @test all(apply(*, UnitUpperTriangular(A)', b) ≈ UnitUpperTriangular(A)'b)
                @test all(apply(*, LowerTriangular(A)', b) ≈ LowerTriangular(A)'b)
                @test all(apply(*, UnitLowerTriangular(A)', b) ≈ UnitLowerTriangular(A)'b)

                @test all(apply(*, transpose(UpperTriangular(A)), b) ≈ transpose(UpperTriangular(A))b)
                @test all(apply(*, transpose(UnitUpperTriangular(A)), b) ≈ transpose(UnitUpperTriangular(A))b)
                @test all(apply(*, transpose(LowerTriangular(A)), b) ≈ transpose(LowerTriangular(A))b)
                @test all(apply(*, transpose(UnitLowerTriangular(A)), b) ≈ transpose(UnitLowerTriangular(A))b)

                B = big.(randn(T,100,100))
                
                @test all(apply(*, UpperTriangular(A)', B) ≈ UpperTriangular(A)'B)
                @test all(apply(*, UnitUpperTriangular(A)', B) ≈ UnitUpperTriangular(A)'B)
                @test all(apply(*, LowerTriangular(A)', B) ≈ LowerTriangular(A)'B)
                @test all(apply(*, UnitLowerTriangular(A)', B) ≈ UnitLowerTriangular(A)'B)

                @test all(apply(*, transpose(UpperTriangular(A)), B) ≈ transpose(UpperTriangular(A))B)
                @test all(apply(*, transpose(UnitUpperTriangular(A)), B) ≈ transpose(UnitUpperTriangular(A))B)
                @test all(apply(*, transpose(LowerTriangular(A)), B) ≈ transpose(LowerTriangular(A))B)
                @test all(apply(*, transpose(UnitLowerTriangular(A)), B) ≈ transpose(UnitLowerTriangular(A))B)                
            end
        end
    end

    @testset "tri Rmul" begin
        for T in (Float64, ComplexF64)
            A = randn(T, 100,100)
            B = randn(T, 100,100)
            R = Rmul(copy(A), UpperTriangular(B))
            @test size(R) == (size(R,1),size(R,2)) == (100,100)
            @test axes(R) == (axes(R,1),axes(R,2)) == (Base.OneTo(100),Base.OneTo(100))
            @test eltype(R) == T
            @test length(R) == 100^2

            @test similar(R) isa Matrix{T}
            @test similar(R,Int) isa Matrix{Int}
            
            @test applied(*, A, UpperTriangular(B)) isa Applied{RmulStyle}
            @test similar(applied(*, A, UpperTriangular(B)), Float64) isa Matrix{Float64}

            R2 = deepcopy(R)
            Ap = applied(*, copy(A), UpperTriangular(B))
            Ap2 = applied(*, copy(A), UpperTriangular(B))
            @test all(BLAS.trmm('R', 'U', 'N', 'N', one(T), B, A) .=== apply(*, A, UpperTriangular(B)) .=== 
                    copyto!(similar(Ap),Ap) .=== materialize!(Ap2) .=== copyto!(similar(R2), R2) .=== materialize!(R))
            @test R.A ≠ A
            @test all(BLAS.trmm('R', 'U', 'T', 'N', one(T), B, A) .=== apply(*, A, transpose(UpperTriangular(B))) .=== A*transpose(UpperTriangular(B)))
            @test all(BLAS.trmm('R', 'U', 'N', 'U', one(T), B, A) .=== apply(*, A, UnitUpperTriangular(B)) .=== A*UnitUpperTriangular(B))
            @test all(BLAS.trmm('R', 'U', 'T', 'U', one(T), B, A) .=== apply(*, A, transpose(UnitUpperTriangular(B))) .=== A*transpose(UnitUpperTriangular(B)))
            @test all(BLAS.trmm('R', 'L', 'N', 'N', one(T), B, A) .=== apply(*, A, LowerTriangular(B)) .=== A*LowerTriangular(B))
            @test all(BLAS.trmm('R', 'L', 'T', 'N', one(T), B, A) .=== apply(*, A, transpose(LowerTriangular(B))) .=== A*transpose(LowerTriangular(B)))
            @test all(BLAS.trmm('R', 'L', 'N', 'U', one(T), B, A) .=== apply(*, A, UnitLowerTriangular(B)) .=== A*UnitLowerTriangular(B))
            @test all(BLAS.trmm('R', 'L', 'T', 'U', one(T), B, A) .=== apply(*, A, transpose(UnitLowerTriangular(B))) .=== A*transpose(UnitLowerTriangular(B)))

            if T <: Complex
                @test all(BLAS.trmm('R', 'U', 'C', 'N', one(T), B, A) .=== apply(*, A, UpperTriangular(B)') .=== A*UpperTriangular(B)')
                @test all(BLAS.trmm('R', 'U', 'C', 'U', one(T), B, A) .=== apply(*, A, UnitUpperTriangular(B)') .=== A*UnitUpperTriangular(B)')
                @test all(BLAS.trmm('R', 'L', 'C', 'N', one(T), B, A) .=== apply(*, A, LowerTriangular(B)') .=== A*LowerTriangular(B)')
                @test all(BLAS.trmm('R', 'L', 'C', 'U', one(T), B, A) .=== apply(*, A, UnitLowerTriangular(B)') .=== A*UnitLowerTriangular(B)')
            end
        end

        T = Float64
        A = big.(randn(100,100))
        B = big.(randn(100,100))
        materialize!(Rmul(A,UpperTriangular(B)))
    end

    @testset "Diagonal and SymTridiagonal" begin
        A = randn(5,5)
        B = Diagonal(randn(5))
        @test MemoryLayout(typeof(B)) == DiagonalLayout{DenseColumnMajor}()
        @test ApplyStyle(*, typeof(A), typeof(B)) == RmulStyle()
        @test apply(*,A,B) == A*B == materialize!(Rmul(copy(A),B))

        @test ApplyStyle(*, typeof(B), typeof(A)) == LmulStyle()
        @test apply(*,B,A) == B*A

        @test ApplyStyle(*, typeof(B), typeof(B)) == LmulStyle()
        @test apply(*,B,B) == B*B
        @test apply(*,B,B) isa Diagonal

        A = randn(5,5)
        B = SymTridiagonal(randn(5),randn(4))
        @test MemoryLayout(typeof(B)) == SymTridiagonalLayout{DenseColumnMajor}()
        @test apply(*,A,B) ≈ A*B
    end
end

@testset "Factorizations" begin
    @testset "QR" begin
        A = randn(5,3)
        b = randn(3)
        B = randn(3,3)
        Q,R = qr(A)
        @test MemoryLayout(typeof(Q)) isa QRCompactWYQLayout
        @test all(Q*b .=== apply(*,Q,b))
        @test all(Q*B .=== copyto!(similar(B,5,3),Lmul(Q,B)) .=== copyto!(similar(B,5,3),applied(*,Q,B)) .=== apply(*,Q,B))
        @test all(Q*B*b .=== apply(*,Q,B,b))
        @test_throws DimensionMismatch apply(*, Q, randn(4))
        @test_throws DimensionMismatch apply(*, Q, randn(4,3))
        dest = fill(NaN,5)
        @test copyto!(dest, applied(*,Q,b)) == Q*b

        b = randn(5)
        B = randn(5,5)
        @test all(Q*b .=== apply(*,Q,b))
        @test all(Q*b .=== apply(*,Q,b))
        @test all(Q*B .=== apply(*,Q,B))
        @test all(Q*B*b .=== apply(*,Q,B,b))

        @test MemoryLayout(typeof(Q')) isa AdjQRCompactWYQLayout
        @test all(Q'b .=== apply(*,Q',b))
        @test all(Q'B .=== apply(*,Q',B))
        @test all(Q'B*b .=== apply(*,Q',B,b))
        @test_throws DimensionMismatch apply(*,Q',randn(3))
        @test_throws DimensionMismatch apply(*,Q',randn(3,3))
        dest = fill(NaN,5)
        @test all(copyto!(dest, applied(*,Q',b)) .=== Q'*b)
    end

    @testset "applied axes" begin
        T = Float64
        A = randn(5,5)
        x = randn(5)
        M =  applied(+, applied(*,T(1.0),A,x), applied(*,T(0.0),x))
        @test axes(M) == axes(M.args[1]) == (Base.OneTo(5),)
    end

    @testset "Broadcast" begin
        A = randn(5,5)
        b = randn(5)
        M = Mul(A,b)
        @test Base.BroadcastStyle(typeof(M)) isa ApplyArrayBroadcastStyle{1}
        @test M .+ 1 ≈ A*b .+ 1

        @test Base.BroadcastStyle(ApplyArrayBroadcastStyle{1}(), Broadcast.DefaultArrayStyle{1}()) == Broadcast.DefaultArrayStyle{1}()
        @test Base.BroadcastStyle(ApplyArrayBroadcastStyle{1}(), Broadcast.DefaultArrayStyle{2}()) == Broadcast.DefaultArrayStyle{2}()
    end

    @testset "Diagonal" begin
       @test Diagonal(Fill(2,10))  * Fill(3,10) ≡ Fill(6,10)
       @test apply(*, Diagonal(Fill(2,10)), Fill(3,10)) ≡ Fill(6,10)
       @test_broken Diagonal(Fill(2,10))  * Fill(3,10,3) ≡ Fill(6,10)
       @test apply(*, Diagonal(Fill(2,10)), Fill(3,10,3)) ≡ Fill(6,10,3)
       @test apply(*,Fill(3,10,10),Fill(3,10)) ≡ Fill(9,10)
       @test apply(*, Eye(10), Ones(10)) == Ones(10)
       @test apply(*, Eye(10), Eye(10)) == Eye(10)
    end

    @testset "ApplyArray MulTest" begin
        A = ApplyArray(*,randn(2,2), randn(2,2))
        @test ApplyStyle(*,typeof(A),typeof(randn(2,2))) == FlattenMulStyle()
        @test ApplyArray(*,Diagonal(Fill(2,10)), Fill(3,10,10))*Fill(3,10) ≡ Fill(18,10)
        @test ApplyArray(*,Diagonal(Fill(2,10)), Fill(3,10,10))*ApplyArray(*,Diagonal(Fill(2,10)), Fill(3,10,10)) == Fill(36,10,10)
    end
end

@testset "MulAdd" begin
    A = randn(5,5)
    B = randn(5,4)
    C = randn(5,4)
    b = randn(5)
    c = randn(5)

    M = MulAdd(2.0,A,B,3.0,C)
    @test size(M) == size(C)
    @test size(M,1) == size(C,1)
    @test size(M,2) == size(C,2)
    @test_broken size(M,3) == size(C,3)
    @test length(M) == length(C)
    @test axes(M) == axes(C)
    @test eltype(M) == Float64
    @test materialize(M) ≈ 2.0A*B + 3.0C

    @test_throws DimensionMismatch materialize(MulAdd(2.0,A,randn(3),1.0,B))
    @test_throws DimensionMismatch materialize(MulAdd([1,2],A,B,[1,2],C))
    @test_throws DimensionMismatch materialize(MulAdd(2.0,A,B,3.0,randn(3,4)))
    @test_throws DimensionMismatch materialize(MulAdd(2.0,A,B,3.0,randn(5,5)))

    B = randn(5,5)
    C = randn(5,5)
    @test materialize(MulAdd(2.0,Diagonal(A),Diagonal(B),3.0,Diagonal(C))) == 
          materialize!(MulAdd(2.0,Diagonal(A),Diagonal(B),3.0,Diagonal(copy(C)))) == 2.0Diagonal(A)*Diagonal(B) + 3.0*Diagonal(C)
    @test_broken materialize(MulAdd(2.0,Diagonal(A),Diagonal(B),3.0,Diagonal(C))) isa Diagonal

    @test materialize(MulAdd(1.0, Eye(5), A, 3.0, C)) == materialize!(MulAdd(1.0, Eye(5), A, 3.0, copy(C))) == A + 3.0C
    @test materialize(MulAdd(1.0, A, Eye(5), 3.0, C)) == materialize!(MulAdd(1.0, A, Eye(5), 3.0, copy(C))) == A + 3.0C
end

@testset "MulArray" begin
  @testset "Basic" begin
        A = randn(5,5)
        M = ApplyArray(*,A,A)
        @test M[5] == M[5,1]
        @test M[6] == M[1,2]
        @test Matrix(M) ≈ A^2
        @test M^2 ≈ A^4
        @test M^2 isa ApplyMatrix{Float64,typeof(*)}
        x = randn(5)
        @test x'M ≈ transpose(x)*M ≈ x'Matrix(M)
        @test_throws DimensionMismatch materialize(applied(*, randn(5,5), randn(4)))
        @test_throws DimensionMismatch ApplyArray(*, randn(5,5), randn(4))
    end

    @testset "Bug in getindex" begin
        M = ApplyArray(*,[1,2,3],Ones(1,20))
        @test M[1,1] == 1
        @test M[2,1] == 2
        M = Applied(*,[1 2; 3 4], [1 2; 3 4])
        @test M[1] == 7
    end

    @testset "Views" begin
        A = randn(500,500)
        b = randn(500)
        M = ApplyArray(*,A,b)

        V = view(M,2:300)
        @test MemoryLayout(typeof(V)) isa ApplyLayout{typeof(*)}
        @test arguments(V) == (view(A,2:300,Base.OneTo(500)),view(b, Base.OneTo(500)))
        @test Applied(V) isa Applied{MulAddStyle}
        @test ApplyArray(V) ≈ (A*b)[2:300]
        c = similar(V)
        copyto!(c,Applied(V))
        VERSION ≥ v"1.2" && @test @allocated(copyto!(c,Applied(V))) ≤ 200
        copyto!(c, V)
        VERSION ≥ v"1.2" && @test @allocated(copyto!(c, V)) ≤ 200
        @test all(c .=== apply(*, arguments(V)...))

        B = randn(500,500)
        M = ApplyArray(*,A,B)
        V = view(M,2:300,3:400)
        @test MemoryLayout(typeof(V)) isa ApplyLayout{typeof(*)}
        @test arguments(V) == (view(A,2:300,Base.OneTo(500)),view(B, Base.OneTo(500),3:400))
        @test Applied(V) isa Applied{MulAddStyle}
        c = similar(V)
        copyto!(c,Applied(V))
        VERSION ≥ v"1.2" && @test @allocated(copyto!(c,Applied(V))) ≤ 1000
        copyto!(c, V)
        VERSION ≥ v"1.2" && @test @allocated(copyto!(c, V)) ≤ 1000
        @test all(c .=== apply(*, arguments(V)...))
    end

    @testset "* algebra" begin
        A = ApplyArray(*,[1 2; 3 4], Vcat(Fill(1,1,3),Fill(2,1,3)))
        @test 2.0A isa ApplyArray
        @test 2.0\A isa ApplyArray
        @test A/2 isa ApplyArray
        @test (2.0A) == 2.0Array(A)
        @test (2.0\A) == 2.0\Array(A)
        @test A/2.0 == Array(A)/2.0
    end

    @testset "* trans" begin
        A = ApplyArray(*,[1 2; 3 4], Vcat(Fill(1,1,3),Fill(2,1,3)))
        V = view(A,1:2,1:2)
        @test MemoryLayout(typeof(V)) isa ApplyLayout{typeof(*)}
        @test V == ApplyArray(V)
        @test MemoryLayout(typeof(V')) isa ApplyLayout{typeof(*)}
        @test V' == ApplyArray(V')
        @test MemoryLayout(typeof(transpose(V))) isa ApplyLayout{typeof(*)}
        @test transpose(V) == ApplyArray(transpose(V))
    end

    @testset "row-vec * fix" begin
        A = ApplyArray(*,[1 2; 3 4], Vcat(Fill(1,1,3),Fill(2,1,3)))
        V = view(A, 2, 1:2)
        @test arguments(V) == ([1 2; 1 2], [3,4])
        @test ApplyArray(V) == Array(V) == A[2,1:2] == Array(A)[2,1:2]

        V = view(A, 1:2, 2)
        @test arguments(V) == ([1 2; 3 4], [1,2])
        @test ApplyArray(V) == Array(V) == A[1:2,2] == Array(A)[1:2,2]
    end

    @testset "argument type inferrence" begin
        n = 10
        L = ApplyArray(*,fill(3.0,n), ones(1,n))
        A,B = @inferred(arguments(L))
        V = view(L,1:3,1:3)
        a,b = @inferred(arguments(V))
        @test a == view(A,1:3,Base.OneTo(1))
        @test b == view(B,Base.OneTo(1),1:3)
    end

    @testset "Mul colsupport" begin
        D = ApplyArray(*,Diagonal(randn(5)),Diagonal(randn(5)),Diagonal(randn(5)))
        D̃ = Applied(*,Diagonal(randn(5)),Diagonal(randn(5)),Diagonal(randn(5)))
        @test colsupport(D,3) == colsupport(D̃,3) == 3:3
        @test rowsupport(D,3) == rowsupport(D̃,3) == 3:3
    end
end