using Test, LinearAlgebra, LazyArrays, StaticArrays, FillArrays


@testset "Mul" begin
    @testset "eltype" begin
        @inferred(eltype(Mul(zeros(Int,2,2), zeros(Float64,2)))) == Float64
        @inferred(eltype(Mul(zeros(ComplexF16,2,2),zeros(Int,2,2),zeros(Float64,2)))) == ComplexF64

        v = Mul(zeros(Int,2,2), zeros(Float64,2))
        A = Mul(zeros(Int,2,2), zeros(Float64,2,2))
        @test @inferred(axes(v)) == (@inferred(axes(v,1)),) == (Base.OneTo(2),)
        @test @inferred(size(v)) == (@inferred(size(v,1)),) == (2,)
        @test @inferred(axes(A)) == (@inferred(axes(A,1)),@inferred(axes(A,2))) == (Base.OneTo(2),Base.OneTo(2))
        @test @inferred(size(A)) == (@inferred(size(A,1)),size(A,2)) == (2,2)
    end

    @testset "gemv Float64" begin
        for A in (randn(5,5), view(randn(5,5),:,:), view(randn(5,5),1:5,:),
                  view(randn(5,5),1:5,1:5), view(randn(5,5),:,1:5)),
            b in (randn(5), view(randn(5),:), view(randn(5),1:5), view(randn(9),1:2:9))
            c = similar(b);

            c .= Mul(A,b)
            @test all(c .=== A*b .=== BLAS.gemv!('N', 1.0, A, b, 0.0, similar(c)))

            copyto!(c, Mul(A,b))
            @test all(c .=== A*b .=== BLAS.gemv!('N', 1.0, A, b, 0.0, similar(c)))

            b̃ = copy(b)
            copyto!(b̃, Mul(A,b̃))
            @test all(c .=== b̃)

            c .= 2.0 .* Mul(A,b)
            @test all(c .=== BLAS.gemv!('N', 2.0, A, b, 0.0, similar(c)))

            c = copy(b)
            c .= Mul(A,b) .+ c
            @test all(c .=== BLAS.gemv!('N', 1.0, A, b, 1.0, copy(b)))

            c = copy(b)
            c .= Mul(A,b) .+ 2.0 .* c
            @test all(c .=== BLAS.gemv!('N', 1.0, A, b, 2.0, copy(b)))

            c = copy(b)
            c .= 2.0 .* Mul(A,b) .+ c
            @test all(c .=== BLAS.gemv!('N', 2.0, A, b, 1.0, copy(b)))

            c = copy(b)
            c .= 3.0 .* Mul(A,b) .+ 2.0 .* c
            @test all(c .=== BLAS.gemv!('N', 3.0, A, b, 2.0, copy(b)))

            d = similar(c)
            c = copy(b)
            d .= 3.0 .* Mul(A,b) .+ 2.0 .* c
            @test all(d .=== BLAS.gemv!('N', 3.0, A, b, 2.0, copy(b)))
        end
    end

    @testset "gemv mixed array types" begin
        (A, b, c) = (randn(5,5), randn(5), 1.0:5.0)
        d = similar(b)
        d .= Mul(A,b) .+ c
        @test all(d .=== BLAS.gemv!('N', 1.0, A, b, 1.0, Vector{Float64}(c)))

        d .= Mul(A,b) .+ 2.0 .* c
        @test all(d .=== BLAS.gemv!('N', 1.0, A, b, 2.0, Vector{Float64}(c)))

        d .= 2.0 .* Mul(A,b) .+ c
        @test all(d .=== BLAS.gemv!('N', 2.0, A, b, 1.0, Vector{Float64}(c)))

        d .= 3.0 .* Mul(A,b) .+ 2.0 .* c
        @test all(d .=== BLAS.gemv!('N', 3.0, A, b, 2.0, Vector{Float64}(c)))

        @test (similar(b) .= 1 .* Mul(A,b)) ≈
            (similar(b) .= 1 .* Mul(A,b) .+ 0 .* b) ≈
            A*b
    end


    @testset "gemv Complex" begin
        for T in (ComplexF64,),
                A in (randn(T,5,5), view(randn(T,5,5),:,:)),
                b in (randn(T,5), view(randn(T,5),:))
            c = similar(b);

            c .= Mul(A,b)
            @test all(c .=== BLAS.gemv!('N', one(T), A, b, zero(T), similar(c)))

            b .= Mul(A,b)
            @test all(c .=== b)

            c .= 2one(T) .* Mul(A,b)
            @test all(c .=== BLAS.gemv!('N', 2one(T), A, b, zero(T), similar(c)))

            c = copy(b)
            c .= Mul(A,b) .+ c
            @test all(c .=== BLAS.gemv!('N', one(T), A, b, one(T), copy(b)))


            c = copy(b)
            c .= Mul(A,b) .+ 2one(T) .* c
            @test all(c .=== BLAS.gemv!('N', one(T), A, b, 2one(T), copy(b)))

            c = copy(b)
            c .= 2one(T) .* Mul(A,b) .+ c
            @test all(c .=== BLAS.gemv!('N', 2one(T), A, b, one(T), copy(b)))


            c = copy(b)
            c .= 3one(T) .* Mul(A,b) .+ 2one(T) .* c
            @test all(c .=== BLAS.gemv!('N', 3one(T), A, b, 2one(T), copy(b)))

            d = similar(c)
            c = copy(b)
            d .= 3one(T) .* Mul(A,b) .+ 2one(T) .* c
            @test all(d .=== BLAS.gemv!('N', 3one(T), A, b, 2one(T), copy(b)))
        end
    end

    @testset "gemm" begin
        for A in (randn(5,5), view(randn(5,5),:,:), view(randn(5,5),1:5,:),
                  view(randn(5,5),1:5,1:5), view(randn(5,5),:,1:5)),
            B in (randn(5,5), view(randn(5,5),:,:), view(randn(5,5),1:5,:),
                      view(randn(5,5),1:5,1:5), view(randn(5,5),:,1:5))
            C = similar(B);

            C .= Mul(A,B)
            @test all(C .=== BLAS.gemm!('N', 'N', 1.0, A, B, 0.0, similar(C)))

            B .= Mul(A,B)
            @test all(C .=== B)

            C .= 2.0 .* Mul(A,B)
            @test all(C .=== BLAS.gemm!('N', 'N', 2.0, A, B, 0.0, similar(C)))

            C = copy(B)
            C .= Mul(A,B) .+ C
            @test all(C .=== BLAS.gemm!('N', 'N', 1.0, A, B, 1.0, copy(B)))


            C = copy(B)
            C .= Mul(A,B) .+ 2.0 .* C
            @test all(C .=== BLAS.gemm!('N', 'N', 1.0, A, B, 2.0, copy(B)))

            C = copy(B)
            C .= 2.0 .* Mul(A,B) .+ C
            @test all(C .=== BLAS.gemm!('N', 'N', 2.0, A, B, 1.0, copy(B)))


            C = copy(B)
            C .= 3.0 .* Mul(A,B) .+ 2.0 .* C
            @test all(C .=== BLAS.gemm!('N', 'N', 3.0, A, B, 2.0, copy(B)))

            d = similar(C)
            C = copy(B)
            d .= 3.0 .* Mul(A,B) .+ 2.0 .* C
            @test all(d .=== BLAS.gemm!('N', 'N', 3.0, A, B, 2.0, copy(B)))
        end
    end

    @testset "gemm Complex" begin
        for T in (ComplexF64,),
                A in (randn(T,5,5), view(randn(T,5,5),:,:)),
                B in (randn(T,5,5), view(randn(T,5,5),:,:))
            C = similar(B);

            C .= Mul(A,B)
            @test all(C .=== BLAS.gemm!('N', 'N', one(T), A, B, zero(T), similar(C)))

            B .= Mul(A,B)
            @test all(C .=== B)

            C .= 2one(T) .* Mul(A,B)
            @test all(C .=== BLAS.gemm!('N', 'N', 2one(T), A, B, zero(T), similar(C)))

            C = copy(B)
            C .= Mul(A,B) .+ C
            @test all(C .=== BLAS.gemm!('N', 'N', one(T), A, B, one(T), copy(B)))


            C = copy(B)
            C .= Mul(A,B) .+ 2one(T) .* C
            @test all(C .=== BLAS.gemm!('N', 'N', one(T), A, B, 2one(T), copy(B)))

            C = copy(B)
            C .= 2one(T) .* Mul(A,B) .+ C
            @test all(C .=== BLAS.gemm!('N', 'N', 2one(T), A, B, one(T), copy(B)))


            C = copy(B)
            C .= 3one(T) .* Mul(A,B) .+ 2one(T) .* C
            @test all(C .=== BLAS.gemm!('N', 'N', 3one(T), A, B, 2one(T), copy(B)))

            d = similar(C)
            C = copy(B)
            d .= 3one(T) .* Mul(A,B) .+ 2one(T) .* C
            @test all(d .=== BLAS.gemm!('N', 'N', 3one(T), A, B, 2one(T), copy(B)))
        end
    end

    @testset "gemv mixed array types" begin
        (A, B, C) = (randn(5,5), randn(5,5), reshape(1.0:25.0,5,5))
        D = similar(B)
        D .= Mul(A,B) .+ C
        @test all(D .=== BLAS.gemm!('N', 'N', 1.0, A, B, 1.0, Matrix{Float64}(C)))

        D .= Mul(A,B) .+ 2.0 .* C
        @test all(D .=== BLAS.gemm!('N', 'N', 1.0, A, B, 2.0, Matrix{Float64}(C)))

        D .= 2.0 .* Mul(A,B) .+ C
        @test all(D .=== BLAS.gemm!('N', 'N', 2.0, A, B, 1.0, Matrix{Float64}(C)))

        D .= 3.0 .* Mul(A,B) .+ 2.0 .* C
        @test all(D .=== BLAS.gemm!('N', 'N', 3.0, A, B, 2.0, Matrix{Float64}(C)))

        @test (similar(B) .= 1 .* Mul(A,B)) ≈
            (similar(B) .= 1 .* Mul(A,B) .+ 0 .* B) ≈
            A*B
    end

    @testset "gemv adjtrans" begin
        for A in (randn(5,5), view(randn(5,5),:,:), view(randn(5,5),1:5,:),
                  view(randn(5,5),1:5,1:5), view(randn(5,5),:,1:5)),
                    b in (randn(5), view(randn(5),:), view(randn(5),1:5), view(randn(9),1:2:9)),
                    Ac in (transpose(A), A')
            c = similar(b);

            c .= Mul(Ac,b)
            @test all(c .=== BLAS.gemv!('T', 1.0, A, b, 0.0, similar(c)))

            b .= Mul(Ac,b)
            @test all(c .=== b)

            c .= 2.0 .* Mul(Ac,b)
            @test all(c .=== BLAS.gemv!('T', 2.0, A, b, 0.0, similar(c)))

            c = copy(b)
            c .= Mul(Ac,b) .+ c
            @test all(c .=== BLAS.gemv!('T', 1.0, A, b, 1.0, copy(b)))


            c = copy(b)
            c .= Mul(Ac,b) .+ 2.0 .* c
            @test all(c .=== BLAS.gemv!('T', 1.0, A, b, 2.0, copy(b)))

            c = copy(b)
            c .= 2.0 .* Mul(Ac,b) .+ c
            @test all(c .=== BLAS.gemv!('T', 2.0, A, b, 1.0, copy(b)))


            c = copy(b)
            c .= 3.0 .* Mul(Ac,b) .+ 2.0 .* c
            @test all(c .=== BLAS.gemv!('T', 3.0, A, b, 2.0, copy(b)))

            d = similar(c)
            c = copy(b)
            d .= 3.0 .* Mul(Ac,b) .+ 2.0 .* c
            @test all(d .=== BLAS.gemv!('T', 3.0, A, b, 2.0, copy(b)))
        end
    end

    @testset "gemv adjtrans mixed types" begin
        (A, b, c) = (randn(5,5), randn(5), 1.0:5.0)
        for Ac in (transpose(A), A')
            d = similar(b)
            d .= Mul(Ac,b) .+ c
            @test all(d .=== BLAS.gemv!('T', 1.0, A, b, 1.0, Vector{Float64}(c)))

            d .= Mul(Ac,b) .+ 2.0 .* c
            @test all(d .=== BLAS.gemv!('T', 1.0, A, b, 2.0, Vector{Float64}(c)))

            d .= 2.0 .* Mul(Ac,b) .+ c
            @test all(d .=== BLAS.gemv!('T', 2.0, A, b, 1.0, Vector{Float64}(c)))

            d .= 3.0 .* Mul(Ac,b) .+ 2.0 .* c
            @test all(d .=== BLAS.gemv!('T', 3.0, A, b, 2.0, Vector{Float64}(c)))
        end
    end

    @testset "gemv adjtrans Complex" begin
        for T in (ComplexF64,),
                A in (randn(T,5,5), view(randn(T,5,5),:,:)),
                b in (randn(T,5), view(randn(T,5),:)),
                (Ac,trans) in ((transpose(A),'T'), (A','C'))
            c = similar(b);

            c .= Mul(Ac,b)
            @test all(c .=== BLAS.gemv!(trans, one(T), A, b, zero(T), similar(c)))

            b .= Mul(Ac,b)
            @test all(c .=== b)

            c .= 2one(T) .* Mul(Ac,b)
            @test all(c .=== BLAS.gemv!(trans, 2one(T), A, b, zero(T), similar(c)))

            c = copy(b)
            c .= Mul(Ac,b) .+ c
            @test all(c .=== BLAS.gemv!(trans, one(T), A, b, one(T), copy(b)))


            c = copy(b)
            c .= Mul(Ac,b) .+ 2one(T) .* c
            @test all(c .=== BLAS.gemv!(trans, one(T), A, b, 2one(T), copy(b)))

            c = copy(b)
            c .= 2one(T) .* Mul(Ac,b) .+ c
            @test all(c .=== BLAS.gemv!(trans, 2one(T), A, b, one(T), copy(b)))


            c = copy(b)
            c .= 3one(T) .* Mul(Ac,b) .+ 2one(T) .* c
            @test all(c .=== BLAS.gemv!(trans, 3one(T), A, b, 2one(T), copy(b)))

            d = similar(c)
            c = copy(b)
            d .= 3one(T) .* Mul(Ac,b) .+ 2one(T) .* c
            @test all(d .=== BLAS.gemv!(trans, 3one(T), A, b, 2one(T), copy(b)))
        end
    end

    @testset "gemm adjtrans" begin
        for A in (randn(5,5), view(randn(5,5),:,:)),
            B in (randn(5,5), view(randn(5,5),1:5,:))
            for Ac in (transpose(A), A')
                C = similar(B)
                C .= Mul(Ac,B)
                @test all(C .=== BLAS.gemm!('T', 'N', 1.0, A, B, 0.0, similar(C)))

                B .= Mul(Ac,B)
                @test all(C .=== B)

                C .= 2.0 .* Mul(Ac,B)
                @test all(C .=== BLAS.gemm!('T', 'N', 2.0, A, B, 0.0, similar(C)))

                C = copy(B)
                C .= Mul(Ac,B) .+ C
                @test all(C .=== BLAS.gemm!('T', 'N', 1.0, A, B, 1.0, copy(B)))


                C = copy(B)
                C .= Mul(Ac,B) .+ 2.0 .* C
                @test all(C .=== BLAS.gemm!('T', 'N', 1.0, A, B, 2.0, copy(B)))

                C = copy(B)
                C .= 2.0 .* Mul(Ac,B) .+ C
                @test all(C .=== BLAS.gemm!('T', 'N', 2.0, A, B, 1.0, copy(B)))


                C = copy(B)
                C .= 3.0 .* Mul(Ac,B) .+ 2.0 .* C
                @test all(C .=== BLAS.gemm!('T', 'N', 3.0, A, B, 2.0, copy(B)))

                d = similar(C)
                C = copy(B)
                d .= 3.0 .* Mul(Ac,B) .+ 2.0 .* C
                @test all(d .=== BLAS.gemm!('T', 'N', 3.0, A, B, 2.0, copy(B)))
            end
            for Bc in (transpose(B), B')
                C = similar(B)
                C .= Mul(A,Bc)
                @test all(C .=== BLAS.gemm!('N', 'T', 1.0, A, B, 0.0, similar(C)))

                B .= Mul(A,Bc)
                @test all(C .=== B)

                C .= 2.0 .* Mul(A,Bc)
                @test all(C .=== BLAS.gemm!('N', 'T', 2.0, A, B, 0.0, similar(C)))

                C = copy(B)
                C .= Mul(A,Bc) .+ C
                @test all(C .=== BLAS.gemm!('N', 'T', 1.0, A, B, 1.0, copy(B)))


                C = copy(B)
                C .= Mul(A,Bc) .+ 2.0 .* C
                @test all(C .=== BLAS.gemm!('N', 'T', 1.0, A, B, 2.0, copy(B)))

                C = copy(B)
                C .= 2.0 .* Mul(A,Bc) .+ C
                @test all(C .=== BLAS.gemm!('N', 'T', 2.0, A, B, 1.0, copy(B)))


                C = copy(B)
                C .= 3.0 .* Mul(A,Bc) .+ 2.0 .* C
                @test all(C .=== BLAS.gemm!('N', 'T', 3.0, A, B, 2.0, copy(B)))

                d = similar(C)
                C = copy(B)
                d .= 3.0 .* Mul(A,Bc) .+ 2.0 .* C
                @test all(d .=== BLAS.gemm!('N', 'T', 3.0, A, B, 2.0, copy(B)))
            end
            for Ac in (transpose(A), A'), Bc in (transpose(B), B')
                C = similar(B)
                C .= Mul(Ac,Bc)
                @test all(C .=== BLAS.gemm!('T', 'T', 1.0, A, B, 0.0, similar(C)))

                B .= Mul(Ac,Bc)
                @test all(C .=== B)

                C .= 2.0 .* Mul(Ac,Bc)
                @test all(C .=== BLAS.gemm!('T', 'T', 2.0, A, B, 0.0, similar(C)))

                C = copy(B)
                C .= Mul(Ac,Bc) .+ C
                @test all(C .=== BLAS.gemm!('T', 'T', 1.0, A, B, 1.0, copy(B)))


                C = copy(B)
                C .= Mul(Ac,Bc) .+ 2.0 .* C
                @test all(C .=== BLAS.gemm!('T', 'T', 1.0, A, B, 2.0, copy(B)))

                C = copy(B)
                C .= 2.0 .* Mul(Ac,Bc) .+ C
                @test all(C .=== BLAS.gemm!('T', 'T', 2.0, A, B, 1.0, copy(B)))


                C = copy(B)
                C .= 3.0 .* Mul(Ac,Bc) .+ 2.0 .* C
                @test all(C .=== BLAS.gemm!('T', 'T', 3.0, A, B, 2.0, copy(B)))

                d = similar(C)
                C = copy(B)
                d .= 3.0 .* Mul(Ac,Bc) .+ 2.0 .* C
                @test all(d .=== BLAS.gemm!('T', 'T', 3.0, A, B, 2.0, copy(B)))
            end
        end
    end

    @testset "symv adjtrans" begin
        A = randn(100,100)
        x = randn(100)

        @test all( (similar(x) .= Mul(Symmetric(A),x)) .===
                    (similar(x) .= Mul(Hermitian(A),x)) .===
                    (similar(x) .= Mul(Symmetric(A)',x)) .===
                    (similar(x) .= Mul(Symmetric(view(A,:,:)',:L),x)) .===
                    (similar(x) .= 1.0.*Mul(Symmetric(A),x) .+ 0.0.*similar(x)) .===
                    (similar(x) .= Mul(view(Symmetric(A),:,:),x)) .===
                    BLAS.symv!('U', 1.0, A, x, 0.0, similar(x)) )

        y = copy(x)
        y .= Mul(Symmetric(A), y)
        @test all( (similar(x) .= Mul(Symmetric(A),x)) .=== y)

        @test all( (similar(x) .= Mul(Symmetric(A,:L),x)) .===
                    (similar(x) .= Mul(Hermitian(A,:L),x)) .===
                    (similar(x) .= Mul(Symmetric(A,:L)',x)) .===
                    (similar(x) .= Mul(Symmetric(view(A,:,:)',:U),x)) .===
                    (similar(x) .= 1.0.*Mul(Symmetric(A,:L),x) .+ 0.0.*similar(x)) .===
                    (similar(x) .= Mul(view(Hermitian(A,:L),:,:),x)) .===
                    BLAS.symv!('L', 1.0, A, x, 0.0, similar(x)) )
        T = ComplexF64
        A = randn(T,100,100)
        x = randn(T,100)
        @test all( (similar(x) .= Mul(Symmetric(A),x)) .===
                    (similar(x) .= Mul(transpose(Symmetric(A)),x)) .===
                    (similar(x) .= Mul(Symmetric(transpose(view(A,:,:)),:L),x)) .===
                    (similar(x) .= one(T).*Mul(Symmetric(A),x) .+ zero(T).*similar(x)) .===
                    (similar(x) .= Mul(view(Symmetric(A),:,:),x)) .===
                    BLAS.symv!('U', one(T), A, x, zero(T), similar(x)) )

        @test all( (similar(x) .= Mul(Symmetric(A,:L),x)) .===
                    (similar(x) .= Mul(transpose(Symmetric(A,:L)),x)) .===
                    (similar(x) .= Mul(Symmetric(transpose(view(A,:,:)),:U),x)) .===
                    (similar(x) .= one(T).*Mul(Symmetric(A,:L),x) .+ zero(T).*similar(x)) .===
                    (similar(x) .= Mul(view(Symmetric(A,:L),:,:),x)) .===
                    BLAS.symv!('L', one(T), A, x, zero(T), similar(x)) )

        @test all( (similar(x) .= Mul(Hermitian(A),x)) .===
                    (similar(x) .= Mul(Hermitian(A)',x)) .===
                    (similar(x) .= one(T).*Mul(Hermitian(A),x) .+ zero(T).*similar(x)) .===
                    (similar(x) .= Mul(view(Hermitian(A),:,:),x)) .===
                    BLAS.hemv!('U', one(T), A, x, zero(T), similar(x)) )

        @test all( (similar(x) .= Mul(Hermitian(A,:L),x)) .===
                    (similar(x) .= Mul(Hermitian(A,:L)',x)) .===
                    (similar(x) .= one(T).*Mul(Hermitian(A,:L),x) .+ zero(T).*similar(x)) .===
                    (similar(x) .= Mul(view(Hermitian(A,:L),:,:),x)) .===
                    BLAS.hemv!('L', one(T), A, x, zero(T), similar(x)) )

        y = copy(x)
        y .= Mul(Hermitian(A), y)
        @test all( (similar(x) .= Mul(Hermitian(A),x)) .=== y)
    end

    @testset "tri" begin
        A = randn(Float64, 100, 100)
        x = randn(Float64, 100)

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

    @testset "Mixed types" begin
        A = randn(5,5)
        b = rand(Int,5)
        c = Array{Float64}(undef, 5)
        c .= Mul(A,b)

        d = similar(c)
        mul!(d, A, b)
        @test all(c .=== d)

        B = rand(Int,5,5)
        C = Array{Float64}(undef, 5, 5)
        C .= Mul(A,B)

        D = similar(C)
        mul!(D, A, B)
        @test all(C .=== D)
    end

    @testset "no allocation" begin
        function blasnoalloc(c, α, A, x, β, y)
            c .= Mul(A,x)
            c .= α .* Mul(A,x)
            c .= Mul(A,x) .+ y
            c .= α .* Mul(A,x) .+ y
            c .= Mul(A,x) .+ β .* y
            c .= α .* Mul(A,x) .+ β .* y
        end

        A = randn(5,5); x = randn(5); y = randn(5); c = similar(y);
        blasnoalloc(c, 2.0, A, x, 3.0, y)
        @test @allocated(blasnoalloc(c, 2.0, A, x, 3.0, y)) == 0
        Ac = A'
        blasnoalloc(c, 2.0, Ac, x, 3.0, y)
        @test @allocated(blasnoalloc(c, 2.0, Ac, x, 3.0, y)) == 0
    end
end
