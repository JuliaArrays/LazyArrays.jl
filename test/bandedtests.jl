module LazyBandedTests
using ArrayLayouts, LazyArrays, BandedMatrices, LinearAlgebra, Test
using BandedMatrices: AbstractBandedLayout, _BandedMatrix, isbanded, BandedStyle, BandedColumns, BandedRows, resize, bandeddata
using LazyArrays: PaddedLayout, PaddedRows, PaddedColumns, arguments, call, LazyArrayStyle, ApplyLayout, simplifiable, resizedata!, MulStyle, LazyLayout, BroadcastLayout
using ArrayLayouts: OnesLayout, StridedLayout
LazyArraysBandedMatricesExt = Base.get_extension(LazyArrays, :LazyArraysBandedMatricesExt)
BroadcastBandedLayout = LazyArraysBandedMatricesExt.BroadcastBandedLayout
ApplyBandedLayout = LazyArraysBandedMatricesExt.ApplyBandedLayout
LazyBandedLayout = LazyArraysBandedMatricesExt.LazyBandedLayout
VcatBandedMatrix = LazyArraysBandedMatricesExt.VcatBandedMatrix

include("mylazyarray.jl")

struct PseudoBandedMatrix{T} <: AbstractMatrix{T}
    data::Array{T}
    l::Int
    u::Int
end

Base.size(A::PseudoBandedMatrix) = size(A.data)
function Base.getindex(A::PseudoBandedMatrix, j::Int, k::Int)
    l, u = bandwidths(A)
    if -l ≤ k-j ≤ u
        A.data[j, k]
    else
        zero(eltype(A.data))
    end
end
function Base.setindex!(A::PseudoBandedMatrix, v, j::Int, k::Int)
    l, u = bandwidths(A)
    if -l ≤ k-j ≤ u
        A.data[j, k] = v
    else
        error("out of band.")
    end
end

struct PseudoBandedLayout <: AbstractBandedLayout end
Base.BroadcastStyle(::Type{<:PseudoBandedMatrix}) = BandedStyle()
BandedMatrices.MemoryLayout(::Type{<:PseudoBandedMatrix}) = PseudoBandedLayout()
BandedMatrices.isbanded(::PseudoBandedMatrix) = true
BandedMatrices.bandwidths(A::PseudoBandedMatrix) = (A.l , A.u)
BandedMatrices.inbands_getindex(A::PseudoBandedMatrix, j::Int, k::Int) = A.data[j, k]
BandedMatrices.inbands_setindex!(A::PseudoBandedMatrix, v, j::Int, k::Int) = setindex!(A.data, v, j, k)
LinearAlgebra.fill!(A::PseudoBandedMatrix, v) = fill!(A.data,v)
ArrayLayouts.lmul!(β::Number, A::PseudoBandedMatrix) = (lmul!(β, A.data); A)
LinearAlgebra.lmul!(β::Number, A::PseudoBandedMatrix) = (lmul!(β, A.data); A)


@testset "Lazy Banded" begin
    @testset "Banded padded" begin
        A = _BandedMatrix((1:10)', 10, -1,1)
        x = Vcat(1:3, Zeros(10-3))
        @test MemoryLayout(x) isa PaddedColumns
        @test A*x isa Vcat{Float64,1,<:Tuple{<:Vector,<:Zeros}}
        @test length((A*x).args[1]) == length(x.args[1]) + bandwidth(A,1) == 2
        @test A*x == A*Vector(x)

        A = _BandedMatrix(randn(3,10), 10, 1,1)
        x = Vcat(randn(10), Zeros(0))
        @test A*x isa Vcat{Float64,1,<:Tuple{<:Vector,<:Zeros}}
        @test length((A*x).args[1]) == 10
        @test A*x ≈ A*Vector(x)

        A = Vcat(Zeros(1,10), brand(9,10,0,2))
        @test MemoryLayout(A) isa ApplyBandedLayout
        @test bandwidths(A) == (1,1)
        @test BandedMatrix(A) == Array(A) == A
        @test A[1:3,1:3] isa BandedMatrix

        A = Hcat(Zeros(5,2), brand(5,5,1,1))
        @test bandwidths(A) == (-1,3)
        @test BandedMatrix(A) == Array(A) == A


        A = Vcat(_BandedMatrix(randn(3,10), 10, 1,1), Vcat(randn(2,10), Zeros(10,10)))
        @test MemoryLayout(A) isa PaddedColumns

        A = Hcat(_BandedMatrix(randn(3,10), 10, 1,1), Hcat(randn(10,2), Zeros(10,10)))
        @test MemoryLayout(A) isa PaddedRows

        A = Hcat(Vcat(1:3, Zeros(7)), _BandedMatrix(randn(3,10), 10, 1,1))
        @test MemoryLayout(A) isa ApplyBandedLayout
    end

    @testset "BroadcastBanded * Padded" begin
        A = BroadcastArray(*, randn(5), brand(5,5,1,2))
        @test axes(A) == (Base.OneTo(5), Base.OneTo(5))
        B = BroadcastArray(*, randn(5,5), brand(5,5,1,2))
        b = Vcat(randn(2), Zeros(3))
        @test A*b ≈ Matrix(A)b
        @test B*b ≈ Matrix(B)b
    end

    @testset "Apply * Banded" begin
        B = brand(5,5,2,1)
        A = ApplyArray(*, B, B)
        @test A * Vcat([1,2], Zeros(3)) ≈ B*B*[1,2,0,0,0]
    end

    @testset "Banded Perturbed" begin
        n = 1000
        D = Diagonal(1:n)
        P = ApplyArray(hvcat, 2, randn(3,3), Zeros(3,n-3), Zeros(n-3,3), Zeros(n-3,n-3))
        @test isbanded(P)
        @test bandwidths(P) == (2,2)

        B = BroadcastArray(+, D, P)
        @test MemoryLayout(B) isa BroadcastBandedLayout
        @test bandwidths(B) == (2,2)

        B = BroadcastArray(+, P, D)
        @test MemoryLayout(B) isa BroadcastBandedLayout
        @test bandwidths(B) == (2,2)

        C = ApplyArray(hvcat, 2, 1, 2, 3, 4)
        @test bandwidths(C) == (1,1)
    end


    @testset "MulMatrix" begin
        @testset "MulBanded" begin
            A = brand(6,5,0,1)
            B = brand(5,5,1,0)

            M = ApplyArray(*, A)
            @test M == A
            @test BandedMatrix(M) == copyto!(similar(A), M) == A

            M = ApplyArray(*,A,B)
            @test isbanded(M) && isbanded(Applied(M))
            @test bandwidths(M) == bandwidths(Applied(M))
            @test BandedMatrix(M) == A*B == copyto!(BandedMatrix(M), M)
            @test MemoryLayout(M) isa ApplyBandedLayout{typeof(*)}
            @test arguments(M) == (A,B)
            @test call(M) == *
            @test colsupport(M,1) == colsupport(Applied(M),1) == 1:2
            @test rowsupport(M,1) == rowsupport(Applied(M),1) == 1:2

            @test Base.BroadcastStyle(typeof(M)) isa LazyArrayStyle{2}
            @test M .+ A ≈ M .+ Matrix(A) ≈ Matrix(A) .+ M

            V = view(M,1:4,1:4)
            @test bandwidths(V) == (1,1)
            @test MemoryLayout(V) == MemoryLayout(M)
            @test M[1:4,1:4] isa BandedMatrix
            @test colsupport(V,1) == 1:2
            @test rowsupport(V,1) == 1:2

            @test MemoryLayout(view(M, [1,3], [2,3])) isa ApplyLayout{typeof(*)}

            A = brand(5,5,0,1)
            B = brand(6,5,1,0)
            @test_throws DimensionMismatch ApplyArray(*,A,B)

            A = brand(6,5,0,1)
            B = brand(5,5,1,0)
            C = brand(5,6,2,2)
            M = applied(*,A,B,C)
            @test @inferred(eltype(M)) == Float64
            @test bandwidths(M) == (3,3)
            @test M[1,1] ≈ (A*B*C)[1,1]

            M = @inferred(ApplyArray(*,A,B,C))
            @test @inferred(eltype(M)) == Float64
            @test bandwidths(M) == (3,3)
            @test BandedMatrix(M) ≈ A*B*C ≈ copyto!(BandedMatrix(M), M)

            M = ApplyArray(*, A, Zeros(5))
            @test colsupport(M,1) == colsupport(Applied(M),1)
            @test colsupport(M,1) == 1:0

            @testset "inv" begin
                A = brand(6,5,0,1)
                B = brand(5,5,1,0)
                C = randn(6,2)
                M = ApplyArray(*,A,B)
                @test M \ C ≈ Matrix(M) \ C
            end

            @testset "Sym/Herm" begin
                A = brand(5,5,0,1)
                B = brand(5,5,1,0)
                M = ApplyArray(*,A,B)
                C = ApplyArray(*,A,im*B)
                @test MemoryLayout(Symmetric(M)) isa SymmetricLayout{LazyBandedLayout}
                @test MemoryLayout(Symmetric(C)) isa SymmetricLayout{LazyBandedLayout}
                @test MemoryLayout(Hermitian(M)) isa SymmetricLayout{LazyBandedLayout}
                @test MemoryLayout(Hermitian(C)) isa HermitianLayout{LazyBandedLayout}
            end
        end

        @testset "Pseudo Mul" begin
            A = PseudoBandedMatrix(rand(5, 4), 1, 2)
            B = PseudoBandedMatrix(rand(4, 4), 2, 3)
            C = PseudoBandedMatrix(zeros(5, 4), 3, 4)
            D = zeros(5, 4)

            @test (C .= applied(*, A, B)) ≈ (D .= applied(*, A, B)) ≈ A*B
        end
        @testset "MulStyle" begin
            A = brand(5,5,0,1)
            B = brand(5,5,1,0)
            C = BroadcastMatrix(*, A, 2)
            M = ApplyArray(*,A,B)
            @test M^2 isa BandedMatrix
            @test M*C isa ApplyMatrix{Float64,typeof(*)}
            @test C*M isa ApplyMatrix{Float64,typeof(*)}
        end
        @testset "Apply*Broadcast" begin
            A = randn(5,5)
            B = randn(5,5)
            C = brand(5,5,1,1)
            D = brand(5,5,1,1)
            @test ApplyArray(*, A, B) * BroadcastArray(*, A, B) ≈ (A*B) * (A .* B)
            @test ApplyArray(*, C, D) * BroadcastArray(*, A, B) ≈ (C*D) * (A .* B)
            @test ApplyArray(*, A, B) * BroadcastArray(*, C, D) ≈ (A*B) * (C .* D)
            @test BroadcastArray(*, A, B) * ApplyArray(*, A, B) ≈ (A .* B) * (A*B)
            @test BroadcastArray(*, C, D) * ApplyArray(*, A, B) ≈ (C .* D) * (A*B)
            @test BroadcastArray(*, A, B) * ApplyArray(*, C, D) ≈ (A .* B) * (C*D)

            @test ApplyArray(*, A, B) \ BroadcastArray(*, A, B) ≈ (A*B) \ (A .* B)
            @test BroadcastArray(*, A, B) \ ApplyArray(*, A, B) ≈ (A .* B) \ (A * B)
            @test BroadcastArray(*, A, B) \ BroadcastArray(*, A, B) ≈ (A .* B) \ (A .* B)
            @test BroadcastArray(*, C, D) \ BroadcastArray(*, C, D) ≈ (C .* D) \ (C .* D)
            @test ApplyArray(*, C, D) \ BroadcastArray(*, C, D) ≈ (C * D) \ (C .* D)
        end

        @testset "flatten" begin
            A = brand(5,5,1,1)
            B = brand(5,5,1,1)
            C = brand(5,5,1,1)
            @test LazyArrays.flatten(ApplyArray(*, A, ApplyArray(*, B, C))) ≈ A * B *C
        end
    end

    @testset "Eye simplifiable" begin
        A = Eye(5)
        B = brand(5,5,1,1)
        C = brand(5,5,1,1)
        D = Diagonal(Fill(2,5))
        @test simplifiable(*, A, BroadcastArray(*, B, C)) == Val(true)
        @test simplifiable(*, BroadcastArray(*, B, C), A) == Val(true)

        @test D * BroadcastArray(*, B, C) ≈ D * (B .* C)
        @test BroadcastArray(*, B, C) * D ≈ (B .* C) * D
    end

    @testset "InvMatrix" begin
        D = brand(5,5,0,0)
        L = brand(5,5,2,0)
        U = brand(5,5,0,2)
        B = brand(5,5,1,2)

        @test bandwidths(ApplyMatrix(inv,D)) == (0,0)
        @test bandwidths(ApplyMatrix(inv,L)) == (4,0)
        @test bandwidths(ApplyMatrix(inv,U)) == (0,4)
        @test bandwidths(ApplyMatrix(inv,B)) == (4,4)

        @test colsupport(ApplyMatrix(inv,D) ,3) == 3:3
        @test colsupport(ApplyMatrix(inv,L), 3) == 3:5
        @test colsupport(ApplyMatrix(inv,U), 3) == 1:3
        @test colsupport(ApplyMatrix(inv,B), 3) == 1:5

        @test rowsupport(ApplyMatrix(inv,D) ,3) == 3:3
        @test rowsupport(ApplyMatrix(inv,L), 3) == 1:3
        @test rowsupport(ApplyMatrix(inv,U), 3) == 3:5
        @test rowsupport(ApplyMatrix(inv,B), 3) == 1:5

        @test bandwidths(ApplyMatrix(\,D,B)) == (1,2)
        @test bandwidths(ApplyMatrix(\,L,B)) == (4,2)
        @test bandwidths(ApplyMatrix(\,U,B)) == (1,4)
        @test bandwidths(ApplyMatrix(\,B,B)) == (4,4)

        @test colsupport(ApplyMatrix(\,D,B), 3) == 1:4
        @test colsupport(ApplyMatrix(\,L,B), 4) == 2:5
        @test colsupport(ApplyMatrix(\,U,B), 3) == 1:4
        @test colsupport(ApplyMatrix(\,B,B), 3) == 1:5

        @test rowsupport(ApplyMatrix(\,D,B), 3) == 2:5
        @test rowsupport(ApplyMatrix(\,L,B), 2) == 1:4
        @test rowsupport(ApplyMatrix(\,U,B), 3) == 2:5
        @test rowsupport(ApplyMatrix(\,B,B), 3) == 1:5

        A = brand(5,5,1,1)
        C = BroadcastMatrix(*, A, 2)
        @test ApplyMatrix(inv,D) * C == inv(D) * (2A)
        @test C * ApplyMatrix(inv,D) == (2A) * inv(D)
        @test ApplyMatrix(inv,D) \ C == D * (2A)
        @test C / ApplyMatrix(inv,D) ≈ (2A) * D

        @test inv(C) ≈ inv(2A)
    end

    @testset "Cat" begin
        A = brand(6,5,2,1)
        H = Hcat(A,A)
        @test H[1,1] == applied(hcat,A,A)[1,1] == A[1,1]
        @test isbanded(H)
        @test bandwidths(H) == (2,6)
        @test BandedMatrix(H) == BandedMatrix(H,(2,6)) == hcat(A,A) == hcat(A,Matrix(A)) ==
                hcat(Matrix(A),A) == hcat(Matrix(A),Matrix(A))
        @test hcat(A,A) isa BandedMatrix
        @test hcat(A,Matrix(A)) isa Matrix
        @test hcat(Matrix(A),A) isa Matrix
        @test isone.(H) isa BandedMatrix
        @test bandwidths(isone.(H)) == (2,6)
        @test @inferred(colsupport(H,1)) == 1:3
        @test Base.replace_in_print_matrix(H,4,1,"0") == "⋅"

        H = Hcat(A,A,A)
        @test isbanded(H)
        @test bandwidths(H) == (2,11)
        @test BandedMatrix(H) == hcat(A,A,A) == hcat(A,Matrix(A),A) == hcat(Matrix(A),A,A) ==
                hcat(Matrix(A),Matrix(A),A) == hcat(Matrix(A),Matrix(A),Matrix(A))
        @test hcat(A,A,A) isa BandedMatrix
        @test isone.(H) isa BandedMatrix
        @test bandwidths(isone.(H)) == (2,11)

        V = Vcat(A,A)
        @test V isa VcatBandedMatrix
        @test isbanded(V)
        @test bandwidths(V) == (8,1)
        @test BandedMatrix(V) == vcat(A,A) == vcat(A,Matrix(A)) == vcat(Matrix(A),A) == vcat(Matrix(A),Matrix(A))
        @test vcat(A,A) isa BandedMatrix
        @test vcat(A,Matrix(A)) isa Matrix
        @test vcat(Matrix(A),A) isa Matrix
        @test isone.(V) isa BandedMatrix
        @test bandwidths(isone.(V)) == (8,1)
        @test Base.replace_in_print_matrix(V,1,3,"0") == "⋅"

        V = Vcat(A,A,A)
        @test bandwidths(V) == (14,1)
        @test BandedMatrix(V) == vcat(A,A,A) == vcat(A,Matrix(A),A) == vcat(Matrix(A),A,A) ==
                vcat(Matrix(A),Matrix(A),A) == vcat(Matrix(A),Matrix(A),Matrix(A))
        @test vcat(A,A,A) isa BandedMatrix
        @test isone.(V) isa BandedMatrix
        @test bandwidths(isone.(V)) == (14,1)
    end

    @testset "BroadcastBanded" begin
        A = BroadcastMatrix(*, brand(5,5,1,2), brand(5,5,2,1))
        @test eltype(A) == Float64
        @test bandwidths(A) == (1,1)
        @test colsupport(A, 1) == 1:2
        @test rowsupport(A, 1) == 1:2
        @test A == broadcast(*, A.args...) == BandedMatrix(A)
        @test MemoryLayout(A) isa BroadcastBandedLayout{typeof(*)}

        B = BandedMatrix{Float64}(undef, (5,5), (1,1))
        @test copyto!(B, A) == B == A

        @test MemoryLayout(A') isa BroadcastBandedLayout{typeof(*)}
        @test bandwidths(A') == (1,1)
        @test colsupport(A',1) == rowsupport(A', 1) == 1:2
        @test A' == BroadcastArray(A') == Array(A)' == BandedMatrix(A')

        V = view(A, 2:3, 3:5)
        @test MemoryLayout(V) isa BroadcastBandedLayout{typeof(*)}
        @test bandwidths(V) == (1,0)
        @test colsupport(V,1) == 1:2
        @test V == BroadcastArray(V) == Array(A)[2:3,3:5]
        @test bandwidths(view(A,2:4,3:5)) == (2,0)

        V = view(A, 2:3, 3:5)'
        @test MemoryLayout(V) isa BroadcastBandedLayout{typeof(*)}
        @test bandwidths(V) == (0,1)
        @test colsupport(V,1) == 1:1
        @test V == BroadcastArray(V) == Array(A)[2:3,3:5]'

        B = BroadcastMatrix(+, brand(5,5,1,2), 2)
        @test B == broadcast(+, B.args...)

        C = BroadcastMatrix(+, brand(5,5,1,2), brand(5,5,3,1))
        @test bandwidths(C) == (3,2)
        @test MemoryLayout(C) == BroadcastBandedLayout{typeof(+)}()
        @test isbanded(C)
        @test BandedMatrix(C) == C == copyto!(BandedMatrix(C), C)

        D = BroadcastMatrix(*, 2, brand(5,5,1,2))
        @test bandwidths(D) == (1,2)
        @test MemoryLayout(D) == BroadcastBandedLayout{typeof(*)}()
        @test isbanded(D)
        @test BandedMatrix(D) == D == copyto!(BandedMatrix(D), D) == 2*D.args[2]

        @testset "band" begin
            @test A[band(0)] == Matrix(A)[band(0)]
            @test B[band(0)] == Matrix(B)[band(0)]
            @test C[band(0)] == Matrix(C)[band(0)]
            @test D[band(0)] == Matrix(D)[band(0)]
        end

        @testset "non-simple" begin
            A = BroadcastMatrix(sin,brand(5,5,1,2))
            @test bandwidths(A) == (1,2)
            @test BandedMatrix(A) == Matrix(A) == A
        end

        @testset "Complex" begin
            C = BroadcastMatrix(*, 2, im*brand(5,5,2,1))
            @test MemoryLayout(C') isa ConjLayout{BroadcastBandedLayout{typeof(*)}}
        end

        @testset "/" begin
            A = BroadcastMatrix(/, brand(5,5,1,2),2)
            B = BandedMatrix{Float64}(undef, size(A), bandwidths(A))
            C = Matrix{Float64}(undef, size(A))
            @test copyto!(B, A) == B == A
            @test copyto!(C, A) == C == A
        end

        @testset "Broadcast *" begin
            A = brand(5,5,1,2)
            B = BroadcastMatrix(*, brand(5,5,1,2), brand(5,5,2,1))
            C = BandedMatrix{Float64}(undef, size(B), (1,2))
            C .= A .+ B
            @test C == A + B
        end

        @testset "generalise broadcast" begin
            for A in (brand(5,5,1,2), Symmetric(brand(5,5,1,2)))
                @test MemoryLayout(BroadcastMatrix(-, A)) isa BroadcastBandedLayout
                @test MemoryLayout(BroadcastMatrix(-, A, A)) isa BroadcastBandedLayout
                @test MemoryLayout(BroadcastMatrix(abs, A)) isa BroadcastBandedLayout
                @test MemoryLayout(BroadcastMatrix(cos, A)) isa BroadcastLayout
            end
        end

        @testset "broadcast_mul_mul" begin
            A = BroadcastMatrix(*, randn(5,5), randn(5,5))
            B = ApplyArray(*, brand(5,5,1,2), brand(5,5,2,1))
            @test A * UpperTriangular(B) ≈ Matrix(A) * UpperTriangular(B)
            @test simplifiable(*, A, UpperTriangular(B)) == Val(false) # TODO: probably should be true
        end
    end

    @testset "Cache" begin
        A = _BandedMatrix(Fill(1,3,10_000), 10_000, 1, 1)
        C = cache(A);
        @test C.data isa BandedMatrix{Int,Matrix{Int},Base.OneTo{Int}}
        @test colsupport(C,1) == rowsupport(C,1) == 1:2
        @test bandwidths(C) == bandwidths(A) == (1,1)
        @test isbanded(C)
        resizedata!(C,1,1);
        @test C[1:10,1:10] == A[1:10,1:10]
        @test C[1:10,1:10] isa BandedMatrix
        @test bandeddata(view(C,1:5,1:5)) == C.data.data[:,1:5]
        @test size(bandeddata(C)) == (3,10000)
    end

    @testset "NaN Bug" begin
        C = BandedMatrix{Float64}(undef, (1,2), (0,2)); C.data .= NaN;
        A = brand(1,1,0,1)
        B = brand(1,2,0,2)
        C .= applied(*, A,B)
        @test C == A*B

        C.data .= NaN
        C .= @~ 1.0 * A*B + 0.0 * C
        @test C == A*B
    end

    @testset "Applied" begin
        A = brand(5,5,1,2)
        @test applied(*,Symmetric(A),A) isa Applied{MulStyle}
        B = apply(*,A,A,A)
        @test B isa BandedMatrix
        @test all(B .=== (A*A)*A)
        @test bandwidths(B) == (3,4)
    end

    @testset "Banded Vcat" begin
        A = Vcat(Zeros(1,10), brand(9,10,1,1))
        @test isbanded(A)
        @test bandwidths(A) == (2,0)
        @test MemoryLayout(A) isa ApplyBandedLayout{typeof(vcat)}
        @test BandedMatrix(A) == Array(A) == A
        @test A*A isa MulMatrix
        @test A*A ≈ BandedMatrix(A)*A ≈ A*BandedMatrix(A) ≈ BandedMatrix(A*A)
        @test A[1:5,1:5] isa BandedMatrix

        A = Vcat(brand(9,10,1,1), Zeros(1,10))
        @test MemoryLayout(A) isa PaddedColumns{<:BandedColumns}
        @test bandwidths(A) == (1,1)
        @test A == BandedMatrix(A)
    end

    @testset "Banded Hcat" begin
        A = Hcat(Zeros(10), brand(10,9,1,1))
        @test isbanded(A)
        @test bandwidths(A) == (0,2)
        @test MemoryLayout(A) isa ApplyBandedLayout{typeof(hcat)}
        @test BandedMatrix(A) == Array(A) == A
        @test A*A isa MulMatrix
        @test A*A ≈ BandedMatrix(A)*A ≈ A*BandedMatrix(A) ≈ BandedMatrix(A*A)
        @test A[1:5,1:5] isa BandedMatrix

        A = Hcat(Zeros(10,3), brand(10,9,1,1))
        @test isbanded(A)
        @test bandwidths(A) == (-2,4)
        @test MemoryLayout(A) isa ApplyBandedLayout{typeof(hcat)}
        @test BandedMatrix(A) == Array(A) == A
        @test A[1:5,1:5] isa BandedMatrix

        A = Hcat(brand(10,9,1,1)', Zeros(9,5))
        @test MemoryLayout(A) isa PaddedRows{<:BandedRows}
        @test bandwidths(A) == (1,1)
        @test BandedMatrix(A) == A
    end

    @testset "Lazy banded * Padded" begin
        A = _BandedMatrix(Vcat(BroadcastArray(exp, 1:5)', Ones(1,5)), 5, 1, 0)
        @test MemoryLayout(A) isa BandedColumns{LazyLayout}
        x = Vcat([1,2], Zeros(3))
        @test A*x isa Vcat
        @test A*A*x isa Vcat

        @test_throws BoundsError muladd!(1.0, A, x, 2.0, Vcat(zeros(2), Zeros(3)))

        y = ones(3)
        @test muladd!(1.0, A, x, 2.0, Vcat(y, Zeros(2))) ≈ A*x .+ [2,2,2,0,0]
        @test y ≈ (A*x .+ 2)[1:3]

        B = PaddedArray(randn(3,4),5,5)
        @test MemoryLayout(A*B) isa PaddedLayout
        @test A*B ≈ Matrix(A)Matrix(B)
        C = BandedMatrix(1 => randn(4))
        @test C*B ≈ Matrix(C)Matrix(B)
        D = BandedMatrix(-1 => randn(4))
        @test D*B ≈ Matrix(D)Matrix(B)

        B.args[2][end,:] .= 0
        C = PaddedArray(randn(3,4),5,5)
        D = deepcopy(C)
        @test muladd!(1.0, A, B, 2.0, D) == D ≈ A*B + 2C
    end

    @testset "Lazy banded" begin
        A = _BandedMatrix(Ones{Int}(1,10),10,0,0)'
        B = _BandedMatrix((-2:-2:-20)', 10,-1,1)
        C = Diagonal( BroadcastVector(/, 2, (1:2:20)))
        C̃ = _BandedMatrix(BroadcastArray(/, 2, (1:2:20)'), 10, -1, 1)
        D = MyLazyArray(randn(10,10))
        M = ApplyArray(*,A,A)
        M̃ = ApplyArray(*,randn(10,10),randn(10,10))
        @test MemoryLayout(A) isa BandedRows{OnesLayout}
        @test MemoryLayout(B) isa BandedColumns{UnknownLayout}
        @test MemoryLayout(C) isa DiagonalLayout{LazyLayout}
        @test MemoryLayout(C̃) isa BandedColumns{LazyLayout}
        BC = BroadcastArray(*, B, permutedims(MyLazyArray(Array(C.diag))))
        @test MemoryLayout(BC) isa BroadcastBandedLayout
        @test A*BC isa MulMatrix
        @test BC*B isa MulMatrix
        @test BC*BC isa MulMatrix
        @test C*C̃ isa MulMatrix
        @test C̃*C isa MulMatrix
        @test C̃*D isa MulMatrix
        @test D*C̃ isa MulMatrix
        @test C̃*M isa MulMatrix
        @test M*C̃ isa MulMatrix
        @test C̃*M̃ isa Matrix
        @test M̃*C̃ isa Matrix

        L = _BandedMatrix(MyLazyArray(randn(3,10)),10,1,1)
        @test Base.BroadcastStyle(typeof(L)) isa LazyArrayStyle{2}

        @test 2 * L isa BandedMatrix
        @test L * 2 isa BandedMatrix
        @test 2 \ L isa BandedMatrix
        @test L / 2 isa BandedMatrix
    end

    @testset "Banded rot" begin
        A = brand(5,5,1,2)
        R = ApplyArray(rot180, A)
        @test MemoryLayout(R) isa BandedColumns{StridedLayout}
        @test bandwidths(R) == (2,1)
        @test BandedMatrix(R) == R == rot180(Matrix(A)) == rot180(A)

        A = brand(5,4,1,2)
        R = ApplyArray(rot180, A)
        @test MemoryLayout(R) isa BandedColumns{StridedLayout}
        @test bandwidths(R) == (3,0)
        @test BandedMatrix(R) == R == rot180(Matrix(A)) == rot180(A)

        A = brand(5,6,1,-1)
        R = ApplyArray(rot180, A)
        @test MemoryLayout(R) isa BandedColumns{StridedLayout}
        @test bandwidths(R) == (-2,2)
        @test BandedMatrix(R) == R == rot180(Matrix(A)) == rot180(A)

        A = brand(6,5,-1,1)
        R = ApplyArray(rot180, A)
        @test MemoryLayout(R) isa BandedColumns{StridedLayout}
        @test bandwidths(R) == (2,-2)
        @test BandedMatrix(R) == R == rot180(Matrix(A)) == rot180(A)

        B = brand(5,4,1,1)
        R = ApplyArray(rot180, ApplyArray(*, A, B))
        @test MemoryLayout(R) isa ApplyBandedLayout{typeof(*)}
        @test bandwidths(R) == (4,-2)
        @test R == rot180(A*B)

        A = brand(5,5,1,2)
        R = ApplyArray(rot180, BroadcastArray(+, A, A))
        @test MemoryLayout(R) isa ApplyBandedLayout{typeof(rot180)}
        @test BandedMatrix(R) == rot180(2A)
    end

    @testset "Triangular bandwidths" begin
        B = brand(5,5,1,2)
        @test bandwidths(ApplyArray(\, Diagonal(randn(5)), B)) == (1,2)
        @test bandwidths(ApplyArray(\, UpperTriangular(randn(5,5)), B)) == (1,4)
        @test bandwidths(ApplyArray(\, LowerTriangular(randn(5,5)), B)) == (4,2)
        @test bandwidths(ApplyArray(\, randn(5,5), B)) == (4,4)
    end

    @testset "zeros mul" begin
        A = _BandedMatrix(BroadcastVector(exp,1:10)', 10, -1,1)
        @test ArrayLayouts.mul(Zeros(5,10),A) ≡ Zeros(5,10)
        @test ArrayLayouts.mul(A,Zeros(10,5)) ≡ Zeros(10,5)
    end

    @testset "concat" begin
        @test MemoryLayout(Vcat(1,1)) isa ApplyLayout{typeof(vcat)}
        @test MemoryLayout(Vcat(1,Zeros(5),1)) isa ApplyLayout{typeof(vcat)}

        @test bandwidths(Hcat(1,randn(1,5))) == (0,5)
        @test bandwidths(Vcat(1,randn(5,1))) == (5,0)

        V = Vcat(brand(5,5,1,1), brand(4,5,0,1))
        @test arguments(view(V,:,1:3)) == (V.args[1][:,1:3], V.args[2][:,1:3])
        H = ApplyArray(hvcat, 2, 1, Hcat(1, Zeros(1,10)), Vcat(1, Zeros(10)), Diagonal(1:11))
        @test bandwidths(H) == (1,1)
        H = ApplyArray(hvcat, 2, 1, Hcat(0, Zeros(1,10)), Vcat(0, Zeros(10)), Diagonal(1:11))
        @test bandwidths(H) == (0,0)
        H = ApplyArray(hvcat, (2,2), 1, Hcat(1, Zeros(1,10)), Vcat(1, Zeros(10)), Diagonal(1:11))
        @test_broken bandwidths(H) == (1,1)

        @test bandwidths(Vcat(Diagonal(1:3), Zeros(3,3))) == (0,0)
        @test bandwidths(Hcat(1, Zeros(1,3))) == (0,0)
        c = cache(Zeros(5));
        @test bandwidths(c) == (-1,0)
    end

    @testset "invlayout * structured banded (#21)" begin
        A = randn(5,5)
        B = BroadcastArray(*, brand(5,5,1,1), 2)
        @test A * B ≈ A * Matrix(B)
        @test A \ B ≈ A \ Matrix(B)
    end


    @testset "QR" begin
        A = brand(100_000,100_000,1,1)
        F = qr(A)
        b = Vcat([1,2,3],Zeros(size(A,1)-3))
        @test F.Q'b == apply(*,F.Q',b)
    end

    @testset "Mul ambiguities (#103)" begin
        A = randn(10,10)
        B = _BandedMatrix(reshape(1:30, 3, 10),10,1,1)
        L = _BandedMatrix(MyLazyArray(randn(3,10)),10,1,1)
        MA = ApplyArray(*,A,A)
        Bi = ApplyArray(inv,B)
        Li = ApplyArray(inv,L)
        M = ApplyArray(*,L,L)

        @test MA * Bi ≈ A*A*Bi
        @test MA * Li ≈ A*A*Li
        @test Bi * MA ≈ Bi*A*A
        @test Li * MA ≈ Li*A*A

        @test Bi * M ≈ inv(B) * L *L
        @test M* Bi ≈ L *L * inv(B)
        @test Li * M ≈ M * Li ≈ L

        @test Li * MA isa Matrix
        @test Li * M isa ApplyArray
        @test M * Li isa ApplyArray

        x = Vcat([1,2], Zeros(8))
        @test Li * x ≈ L \ x
        @test Bi * x ≈ B \ x
    end

    @testset "Banded kron" begin
        @testset "2D" begin
            A = brand(5,5,2,2)
            B = brand(2,2,1,0)
            @test isbanded(Kron(A,B))
            K = kron(A,B)
            @test K isa BandedMatrix
            @test bandwidths(K) == (5,4)
            @test Matrix(K) == kron(Matrix(A), Matrix(B))

            A = brand(3,4,1,1)
            B = brand(3,2,1,0)
            K = kron(A,B)
            @test K isa BandedMatrix
            @test bandwidths(K) == (7,2)
            @test Matrix(K) ≈ kron(Matrix(A), Matrix(B))
            K = kron(B,A)
            @test Matrix(K) ≈ kron(Matrix(B), Matrix(A))

            K = kron(A, B')
            @test K isa BandedMatrix
            @test Matrix(K) ≈ kron(Matrix(A), Matrix(B'))
            K = kron(A', B)
            @test K isa BandedMatrix
            @test Matrix(K) ≈ kron(Matrix(A'), Matrix(B))
            K = kron(A', B')
            @test K isa BandedMatrix
            @test Matrix(K) ≈ kron(Matrix(A'), Matrix(B'))

            A = brand(5,6,2,2)
            B = brand(3,2,1,0)
            K = kron(A,B)
            @test K isa BandedMatrix
            @test bandwidths(K) == (12,4)
            @test Matrix(K) ≈ kron(Matrix(A), Matrix(B))

            n = 10; h = 1/n
            D² = BandedMatrix(0 => Fill(-2,n), 1 => Fill(1,n-1), -1 => Fill(1,n-1))
            D_xx = kron(D², Eye(n))
            D_yy = kron(Eye(n), D²)
            @test D_xx isa BandedMatrix
            @test D_yy isa BandedMatrix
            @test bandwidths(D_xx) == (10,10)
            @test bandwidths(D_yy) == (1,1)
            X = randn(n,n)
            @test reshape(D_xx*vec(X),n,n) ≈ X*D²'
            @test reshape(D_yy*vec(X),n,n) ≈ D²*X
            Δ = D_xx + D_yy
            @test Δ isa BandedMatrix
            @test bandwidths(Δ) == (10,10)
        end

        @testset "#87" begin
            @test kron(Diagonal([1,2,3]), Eye(3)) isa Diagonal{Float64,Vector{Float64}}
        end

        @testset "3D" begin
            n = 10; h = 1/n
            D² = BandedMatrix(0 => Fill(-2,n), 1 => Fill(1,n-1), -1 => Fill(1,n-1))

            D_xx = kron(D², Eye(n), Eye(n))
            D_yy = kron(Eye(n), D², Eye(n))
            D_zz = kron(Eye(n), Eye(n), D²)
            @test bandwidths(D_xx) == (n^2,n^2)
            @test bandwidths(D_yy) == (n,n)
            @test bandwidths(D_zz) == (1,1)

            X = randn(n,n,n)

            Y = similar(X)
            for k = 1:n, j=1:n Y[k,j,:] = D²*X[k,j,:] end
            @test reshape(D_xx*vec(X), n, n, n) ≈ Y
            for k = 1:n, j=1:n Y[k,:,j] = D²*X[k,:,j] end
            @test reshape(D_yy*vec(X), n, n, n) ≈ Y
            for k = 1:n, j=1:n Y[:,k,j] = D²*X[:,k,j] end
            @test reshape(D_zz*vec(X), n, n, n) ≈ Y
        end
    end

    @testset "vec broadcasting" begin
        A = brand(5,5,0,1)
        b = Vcat(1,Zeros(4))
        @test A .+ b == b .+ A == Matrix(A) .+ Vector(b)
    end

    @testset "cache" begin
        B = brand(10,10,2,1)
        A = ApplyArray(*, B, B)
        @test cache(A).data isa BandedMatrix
        C = cache(BandedMatrix,A)
        @test isbanded(C)
        @test bandwidths(C) == (4,2)
        @test C ≈ A
    end

    @testset "I multiplication" begin
        B = brand(10,10,2,1)
        A = ApplyArray(*, B, B)
        @test Eye(10) * A == A * Eye(10) == A
    end

    @testset "Special Mul Overloads" begin
        A = randn(5,5)
        B = brand(5,5,2,1)
        @test ApplyArray(\, A, A) * BroadcastArray(*, B, 3) ≈ 3B
        @test BroadcastArray(*, 2, A) * BroadcastArray(*, B, 3) ≈ 6A*B
    end

    @testset "Lazy data" begin
        D = MyLazyArray(randn(3,5))
        B = _BandedMatrix(D, 5, 1,1)
        A = BroadcastArray(*, 2,  randn(5,5))
        @test B*A ≈ B*Matrix(A)
        @test A*B ≈ Matrix(A)B
        @test simplifiable(*, randn(5,5), B) isa Val{true}
        @test simplifiable(*, B, randn(5,5)) isa Val{true}
    end

    @testset "PaddedRows * Banded" begin
        P = Hcat(randn(2,3), Zeros(2,2))
        B = brand(5,5,2,1)
        @test ApplyArray(*,P,B) ≈ P*B
    end

    @testset "banded hvcat" begin
        n = 10
        B = brand(n-1,n-1,2,1)
        P = ApplyArray(hvcat, 2,
                        1,          Zeros(n-1)',
                        Zeros(n-1), B)
        @test MemoryLayout(P) isa ApplyBandedLayout
        P = ApplyArray(hvcat, 3,
                        1,          2,          Zeros(n-2)',
                        3,          4,          Zeros(n-2)',
                        Zeros(n-2), Zeros(n-2), B)
        @test MemoryLayout(P) isa ApplyBandedLayout
    end

    @testset "dual padded rows" begin
        n = 10
        B = BroadcastArray(*, 3, brand(n,n,2,1))
        a = Vcat(randn(3), Zeros(n-3))
        c = cache(Zeros(n)); c[1] = 2;
        muladd!(2.0, a', B, 3.0, c');
        @test c.datasize == (4,)
        @test c' ≈ 2*a'*B + 3*[2; Zeros(n-1)]'
        @test a'B ≈ a'Matrix(B)
    end

    @testset "lazy banded * Lazy" begin
        n = 10
        A = BroadcastArray(*, 3, brand(n,n,2,1))
        B = BroadcastArray(*, 3, rand(n,n))

        @test A*LowerTriangular(B) ≈ Matrix(A)*LowerTriangular(B)
    end

    @testset "Lazy triangular" begin
        n = 10
        A = BroadcastArray(*, 3, brand(n,n,2,1))
        @test MemoryLayout(UpperTriangular(A)) isa TriangularLayout{'U', 'N', LazyBandedLayout}
        @test MemoryLayout(LowerTriangular(A)) isa TriangularLayout{'L', 'N', LazyBandedLayout}
    end

    @testset "lazy banded Ldiv" begin
        n = 10
        A = BroadcastArray(*, 3, brand(n,n,2,1))
        b = randn(n)
        @test A \ b ≈ Matrix(A) \ b
        @test A \ b isa Vector
        b = Vcat(randn(3), Zeros(7))
        @test inv(A) * b == A \ b
        @test A\b isa Vector

        @test A \ A ≈ I
        @test A \ A isa ApplyArray
        
        M = ApplyArray(*, randn(n,n), b)
        @test A \ M ≈ Matrix(A) \ M
    end
    
    @testset "copyto! broadcast view" begin
        n = 10
        A = brand(n,n,2,1)
        B = BroadcastArray(\, 1:n, A)
        @test B[1:3,1:4] ≈ (1:3) .\ A[1:3,1:4]
        @test B'[1:3,1:4] ≈ B[1:4,1:3]'
    end

    @testset "Ldiv with Diagonal" begin
        n = 10
        a = BroadcastArray(*, 2, 2 .+ rand(n))
        b = BroadcastArray(*, 2, randn(n-1))
        B = Bidiagonal(a, b, :U)
        @test ldiv(B, Diagonal(1:n)) ≈ ApplyArray(inv,B) * Diagonal(1:n) ≈ B \ Diagonal(1:n)
    end
    
    @testset "Ambiguity between ldiv(BandedLazy, Lazy) and ldiv(ApplyBandedLazy, Lazy) (#324)" begin
        A = ApplyArray(*, brand(5, 2, 3), brand(5, 2, 3))
        B = ApplyArray(inv, rand(5, 5))
        @test A \ B ≈ Matrix(A) \ Matrix(B)

        A = Bidiagonal(ApplyVector(+, 1:5, 1:5), ApplyVector(+, 1:4, 1:4), 'U')
        @test A \ B ≈ Matrix(A) \ Matrix(B)
    end

    @testset "Issue #325" begin
        A = cache(reshape(1:25,5,5))
        @test A[band(0)] == [1, 7, 13, 19, 25]
        A = cache(rand(23, 8))
        @test A[band(1)] == [A[i, i+1] for i in 1:7]
        @test A[band(-1)] == [A[i+1, i] for i in 1:8]
    end

    @testset "Banded * Padded" begin
        n = 10
        A = _BandedMatrix(MyLazyArray(randn(3,n)),n,1,1)
        B = brand(n,n,1,1)
        C = BroadcastArray(+, B)
        x = Vcat([1,2,3], Zeros(n-3))
        y = randn(n)
        @test A*x isa Vcat
        @test B*x isa Vcat
        @test C*x isa Vcat
        @test simplifiable(*, A, x) == Val(true)
        @test simplifiable(*, B, x) == Val(true)
        @test simplifiable(*, C, x) == Val(true)

        @test A*y isa Vector
        @test C*y isa Vector
        @test simplifiable(*, A, y) == Val(true)
        @test simplifiable(*, C, y) == Val(true)

        @test x'A isa Adjoint{<:Any,<:Vcat}
        @test x'B isa Adjoint{<:Any,<:Vcat}
        @test x'C isa Adjoint{<:Any,<:Vcat}
        @test transpose(x)A isa Transpose{<:Any,<:Vcat}
        @test transpose(x)B isa Transpose{<:Any,<:Vcat}
        @test transpose(x)C isa Transpose{<:Any,<:Vcat}
        @test simplifiable(*, x', A) == Val(true)
        @test simplifiable(*, x', B) == Val(true)
        @test simplifiable(*, x', C) == Val(true)
        @test simplifiable(*, transpose(x), A) == Val(true)
        @test simplifiable(*, transpose(x), B) == Val(true)
        @test simplifiable(*, transpose(x), C) == Val(true)
        @test x'A ≈ x'Matrix(A)
        @test x'B ≈ x'Matrix(B)
        @test x'C ≈ x'Matrix(C)

        @test y'A isa Adjoint{<:Any,<:Vector}
        @test y'C isa Adjoint{<:Any,<:Vector}
        @test simplifiable(*, y', A) == Val(true)
        @test simplifiable(*, y', B) == Val(true)
        @test simplifiable(*, y', C) == Val(true)
        @test y'A ≈ y'Matrix(A)
        @test y'C ≈ y'Matrix(C)
    end
end

@testset "Issue #329" begin
    # Make sure that we aren't giving the incorrect bandwidths 
    U = UpperTriangular(ApplyArray(inv, brand(5, 5, 1, 2)))
    invU = inv(U)
    L = LowerTriangular(ApplyArray(inv, brand(10, 10, 3, 4)))
    invL = inv(L)
    @test colsupport(invU, 1) == 1:1 
    @test colsupport(invU, 3) == 1:3 
    @test rowsupport(invU, 1) == 1:5 
    @test rowsupport(invU, 4) == 4:5 
    @test rowsupport(invU, 5) == 5:5 
    @test colsupport(invL, 1) == 1:10 
    @test colsupport(invL, 5) == 5:10 
    @test rowsupport(invL, 1) == 1:1 
    @test rowsupport(invL, 4) == 1:4 
end

end # module