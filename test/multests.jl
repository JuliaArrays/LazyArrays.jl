using Test, LinearAlgebra, LazyArrays, StaticArrays, FillArrays
import LazyArrays: MulAdd, MemoryLayout, DenseColumnMajor, DiagonalLayout, SymTridiagonalLayout, Add, AddArray, 
                    MulAddStyle, Applied, ApplyStyle, LmulStyle
import Base.Broadcast: materialize, materialize!, broadcasted

@testset "Mul" begin
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

    @testset "gemv mixed array types" begin
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

    @testset "tri" begin
        @testset "Float * Float vector" begin
            A = randn(Float64, 100, 100)
            x = randn(Float64, 100)

            @test ApplyStyle(*, typeof(UpperTriangular(A)), typeof(x)) isa LmulStyle

            @test all((y = copy(x); y .= Mul(UpperTriangular(A),y) ) .===
                        (similar(x) .= Mul(UpperTriangular(A),x)) .===
                        BLAS.trmv!('U', 'N', 'N', A, copy(x)))
            @test all((y = copy(x); y .= Mul(UnitUpperTriangular(A),y) ) .===
                        (similar(x) .= Mul(UnitUpperTriangular(A),x)) .===
                        BLAS.trmv!('U', 'N', 'U', A, copy(x)))
            @test all((y = copy(x); y .= Mul(LowerTriangular(A),y) ) .===
                        (similar(x) .= Mul(LowerTriangular(A),x)) .===
                        BLAS.trmv!('L', 'N', 'N', A, copy(x)))
            @test all((y = copy(x); y .= Mul(UnitLowerTriangular(A),y) ) .===
                        (similar(x) .= Mul(UnitLowerTriangular(A),x)) .===
                        BLAS.trmv!('L', 'N', 'U', A, copy(x)))

            @test all((y = copy(x); y .= Mul(UpperTriangular(A)',y) ) .===
                        (similar(x) .= Mul(UpperTriangular(A)',x)) .===
                        (similar(x) .= Mul(LowerTriangular(A'),x)) .===
                        BLAS.trmv!('L', 'T', 'N', A, copy(x)))
            @test all((y = copy(x); y .= Mul(UnitUpperTriangular(A)',y) ) .===
                        (similar(x) .= Mul(UnitUpperTriangular(A)',x)) .===
                        (similar(x) .= Mul(UnitLowerTriangular(A'),x)) .===
                        BLAS.trmv!('L', 'T', 'U', A, copy(x)))
            @test all((y = copy(x); y .= Mul(LowerTriangular(A)',y) ) .===
                        (similar(x) .= Mul(LowerTriangular(A)',x)) .===
                        (similar(x) .= Mul(UpperTriangular(A'),x)) .===
                        BLAS.trmv!('U', 'T', 'N', A, copy(x)))
            @test all((y = copy(x); y .= Mul(UnitLowerTriangular(A)',y) ) .===
                        (similar(x) .= Mul(UnitLowerTriangular(A)',x)) .===
                        (similar(x) .= Mul(UnitUpperTriangular(A'),x)) .===
                        BLAS.trmv!('U', 'T', 'U', A, copy(x)))
        end

        @testset "Float * Complex vector"  begin
            T = ComplexF64
            A = randn(T, 100, 100)
            x = randn(T, 100)

            @test all((y = copy(x); y .= Mul(UpperTriangular(A),y) ) .===
                        (similar(x) .= Mul(UpperTriangular(A),x)) .===
                        BLAS.trmv!('U', 'N', 'N', A, copy(x)))
            @test all((y = copy(x); y .= Mul(UnitUpperTriangular(A),y) ) .===
                        (similar(x) .= Mul(UnitUpperTriangular(A),x)) .===
                        BLAS.trmv!('U', 'N', 'U', A, copy(x)))
            @test all((y = copy(x); y .= Mul(LowerTriangular(A),y) ) .===
                        (similar(x) .= Mul(LowerTriangular(A),x)) .===
                        BLAS.trmv!('L', 'N', 'N', A, copy(x)))
            @test all((y = copy(x); y .= Mul(UnitLowerTriangular(A),y) ) .===
                        (similar(x) .= Mul(UnitLowerTriangular(A),x)) .===
                        BLAS.trmv!('L', 'N', 'U', A, copy(x)))
            LowerTriangular(A')  == UpperTriangular(A)'


            @test all((y = copy(x); y .= Mul(transpose(UpperTriangular(A)),y) ) .===
                        (similar(x) .= Mul(transpose(UpperTriangular(A)),x)) .===
                        (similar(x) .= Mul(LowerTriangular(transpose(A)),x)) .===
                        BLAS.trmv!('L', 'T', 'N', A, copy(x)))
            @test all((y = copy(x); y .= Mul(transpose(UnitUpperTriangular(A)),y) ) .===
                        (similar(x) .= Mul(transpose(UnitUpperTriangular(A)),x)) .===
                        (similar(x) .= Mul(UnitLowerTriangular(transpose(A)),x)) .===
                        BLAS.trmv!('L', 'T', 'U', A, copy(x)))
            @test all((y = copy(x); y .= Mul(transpose(LowerTriangular(A)),y) ) .===
                        (similar(x) .= Mul(transpose(LowerTriangular(A)),x)) .===
                        (similar(x) .= Mul(UpperTriangular(transpose(A)),x)) .===
                        BLAS.trmv!('U', 'T', 'N', A, copy(x)))
            @test all((y = copy(x); y .= Mul(transpose(UnitLowerTriangular(A)),y) ) .===
                        (similar(x) .= Mul(transpose(UnitLowerTriangular(A)),x)) .===
                        (similar(x) .= Mul(UnitUpperTriangular(transpose(A)),x)) .===
                        BLAS.trmv!('U', 'T', 'U', A, copy(x)))

            @test all((y = copy(x); y .= Mul(UpperTriangular(A)',y) ) .===
                        (similar(x) .= Mul(UpperTriangular(A)',x)) .===
                        (similar(x) .= Mul(LowerTriangular(A'),x)) .===
                        BLAS.trmv!('L', 'C', 'N', A, copy(x)))
            @test all((y = copy(x); y .= Mul(UnitUpperTriangular(A)',y) ) .===
                        (similar(x) .= Mul(UnitUpperTriangular(A)',x)) .===
                        (similar(x) .= Mul(UnitLowerTriangular(A'),x)) .===
                        BLAS.trmv!('L', 'C', 'U', A, copy(x)))
            @test all((y = copy(x); y .= Mul(LowerTriangular(A)',y) ) .===
                        (similar(x) .= Mul(LowerTriangular(A)',x)) .===
                        (similar(x) .= Mul(UpperTriangular(A'),x)) .===
                        BLAS.trmv!('U', 'C', 'N', A, copy(x)))
            @test all((y = copy(x); y .= Mul(UnitLowerTriangular(A)',y) ) .===
                        (similar(x) .= Mul(UnitLowerTriangular(A)',x)) .===
                        (similar(x) .= Mul(UnitUpperTriangular(A'),x)) .===
                        BLAS.trmv!('U', 'C', 'U', A, copy(x)))
        end

        @testset "Float * Float Matrix" begin
            A = randn(Float64, 100, 100)
            x = randn(Float64, 100, 100)

            @test UpperTriangular(A)*x ≈ (similar(x) .= Mul(UpperTriangular(A), x))
        end
    end

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
        @test all((C .= Mul(A,B)  ) .=== copyto!(similar(C), MulAdd(1.0, A, B, 0.0, C)) .=== A*B)
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
        blasnoalloc(c, 2.0, A, x, 3.0, y)
        @test @allocated(blasnoalloc(c, 2.0, A, x, 3.0, y)) == 0
        Ac = A'
        blasnoalloc(c, 2.0, Ac, x, 3.0, y)
        @test @allocated(blasnoalloc(c, 2.0, Ac, x, 3.0, y)) == 0
        Aa = ApplyArray(+, A, Ac)
        blasnoalloc(c, 2.0, Aa, x, 3.0, y)
        @test_broken @allocated(blasnoalloc(c, 2.0, Aa, x, 3.0, y)) == 0
    end

    @testset "multi-argument mul" begin
        A = randn(5,5)
        B = apply(*,A,A,A)
        @test B isa Matrix{Float64}
        @test all(B .=== (A*A)*A)
    end

    @testset "Diagonal and SymTridiagonal" begin
        A = randn(5,5)
        B = Diagonal(randn(5))
        @test MemoryLayout(typeof(B)) == DiagonalLayout{DenseColumnMajor}()
        @test apply(*,A,B) == A*B

        A = randn(5,5)
        B = SymTridiagonal(randn(5),randn(4))
        @test MemoryLayout(typeof(B)) == SymTridiagonalLayout{DenseColumnMajor}()
        @test apply(*,A,B) == A*B
    end

    @testset "MulArray" begin
        A = randn(5,5)
        M = ApplyArray(*,A,A)
        @test M[5] == M[5,1]
        @test M[6] == M[1,2]
        @test Matrix(M) ≈ A^2
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

    @testset "QR" begin
        A = randn(5,3)
        b = randn(3)
        B = randn(3,3)
        Q,R = qr(A)
        @test all(Q*b .=== apply(*,Q,b))
        @test all(Q*B .=== apply(*,Q,B))
        @test all(Q*B*b .=== apply(*,Q,B,b))
        @test_throws DimensionMismatch apply(*, Q, randn(4))
        @test_throws DimensionMismatch apply(*, Q, randn(4,3))
    end
end



