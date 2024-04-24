module LazyArraysBlockBandedMatricesTest

using LazyArrays, BlockBandedMatrices, BlockArrays, Test
using LinearAlgebra
using ArrayLayouts
using BandedMatrices
using LazyArrays: paddeddata
import BlockArrays: blockcolsupport, blockrowsupport
import LazyArrays: arguments, colsupport, rowsupport,
                    PaddedLayout, paddeddata, ApplyLayout, LazyArrayStyle
import LazyBandedMatrices: BroadcastBlockBandedLayout, BroadcastBandedBlockBandedLayout,
                    ApplyBlockBandedLayout, ApplyBandedBlockBandedLayout


@testset "Block" begin
    @testset "BlockBanded and padded" begin
        A = BlockBandedMatrix{Float64}(undef, 1:4, 1:4, (1,0)); A.data .= randn.();
        D = mortar(Diagonal([randn(k,k) for k=1:4]))
        c = Vcat(randn(3), Zeros(7))
        b = PseudoBlockVector(c, (axes(A,2),))
        @test MemoryLayout(A*b) isa PaddedLayout
        @test MemoryLayout(A*c) isa PaddedLayout
        @test A*b ≈ A*c ≈ Matrix(A)*Vector(b)
        @test D*b ≈ D*c ≈ Matrix(D)*Vector(b)
    end


    @testset "MulBlockBanded" begin
        A = BlockBandedMatrix{Float64}(undef, 1:4, 1:4, (1,0)); A.data .= randn.();
        B = BlockBandedMatrix{Float64}(undef, 1:4, 1:4, (1,1)); B.data .= randn.();
        M = ApplyMatrix(*, A, B)
        @test blockbandwidths(M) == (2,1)
        @test MemoryLayout(M) isa ApplyBlockBandedLayout{typeof(*)}
        @test Base.BroadcastStyle(typeof(M)) isa LazyArrayStyle{2}
        @test BlockBandedMatrix(M) ≈ A*B
        @test arguments(M) == (A,B)
        V = view(M, Block.(1:2), Block.(1:2))
        @test MemoryLayout(V) isa ApplyBlockBandedLayout{typeof(*)}
        @test arguments(V) == (A[Block.(1:2),Block.(1:2)], B[Block.(1:2),Block.(1:2)])
        @test M[Block.(1:2), Block.(1:2)] isa BlockBandedMatrix
    end
    @testset "MulBandedBlockBanded" begin
        A = BandedBlockBandedMatrix{Float64}(undef, 1:4, 1:4, (1,0), (1,0)); A.data .= randn.();
        B = BandedBlockBandedMatrix{Float64}(undef, 1:4, 1:4, (1,1), (1,1)); B.data .= randn.();
        M = ApplyMatrix(*, A, B)
        @test blockbandwidths(M) == (2,1)
        @test subblockbandwidths(M) == (2,1)
        @test MemoryLayout(M) isa ApplyBandedBlockBandedLayout{typeof(*)}
        @test Base.BroadcastStyle(typeof(M)) isa LazyArrayStyle{2}
        @test BandedBlockBandedMatrix(M) ≈ BlockBandedMatrix(M) ≈ A*B
        @test arguments(M) == (A,B)
        V = view(M, Block.(1:2), Block.(1:2))
        @test MemoryLayout(V) isa ApplyBandedBlockBandedLayout{typeof(*)}
        @test arguments(V) == (A[Block.(1:2),Block.(1:2)], B[Block.(1:2),Block.(1:2)])
        @test M[Block.(1:2), Block.(1:2)] isa BandedBlockBandedMatrix
        V = view(M, 1:3, 1:3)
        @test MemoryLayout(V) isa ApplyLayout{typeof(*)}
        @test arguments(V) == (A[1:3,1:3], B[1:3,1:3])
        @test M[1:3, 1:3] ≈ (A*B)[1:3,1:3]

        @test M[Block(2)[1:2],Block(2)[1:2]] isa BandedMatrix
        @test M[Block(2)[1:2],Block(2)] isa BandedMatrix
        @test M[Block(2),Block(2)[1:2]] isa BandedMatrix
        @test M[Block.(1:2), Block.(2:3)] isa BandedBlockBandedMatrix
        @test M[Block(2),Block.(2:3)] isa PseudoBlockArray
        @test M[Block.(2:3),Block(2)] isa PseudoBlockArray
        @test M[Block.(2:3),Block(2)[1:2]] isa PseudoBlockArray
        @test M[Block(2)[1:2],Block.(2:3)] isa PseudoBlockArray
    end

    @testset "BroadcastMatrix" begin
        @testset "BroadcastBlockBanded" begin
            A = BlockBandedMatrix(randn(6,6),1:3,1:3,(1,1))
            B = BroadcastMatrix(*, 2, A)
            @test blockbandwidths(B) == (1,1)
            @test MemoryLayout(B) == BroadcastBlockBandedLayout{typeof(*)}()
            @test BandedBlockBandedMatrix(B) == B == copyto!(BandedBlockBandedMatrix(B), B) == 2*B.args[2]
            @test MemoryLayout(B') isa LazyBandedMatrices.LazyBlockBandedLayout
            @test BlockBandedMatrix(B') == B'

            x = randn(size(B,2))
            @test B*x ≈ 2A*x

            C = BroadcastMatrix(*, 2, im*A)
            @test MemoryLayout(C') isa LazyBandedMatrices.LazyBlockBandedLayout
            @test MemoryLayout(transpose(C)) isa LazyBandedMatrices.LazyBlockBandedLayout

            E = BroadcastMatrix(*, A, 2)
            @test MemoryLayout(E) == BroadcastBlockBandedLayout{typeof(*)}()


            D = Diagonal(PseudoBlockArray(randn(6),1:3))
            @test MemoryLayout(BroadcastMatrix(*, A, D)) isa BroadcastBlockBandedLayout{typeof(*)}
            @test MemoryLayout(BroadcastMatrix(*, D, A)) isa BroadcastBlockBandedLayout{typeof(*)}

            F = BroadcastMatrix(*, A, A)
            @test MemoryLayout(F) == BroadcastBlockBandedLayout{typeof(*)}()
        end
        @testset "BroadcastBandedBlockBanded" begin
            A = BandedBlockBandedMatrix(randn(6,6),1:3,1:3,(1,1),(1,1))

            B = BroadcastMatrix(*, 2, A)
            @test blockbandwidths(B) == (1,1)
            @test subblockbandwidths(B) == (1,1)
            @test MemoryLayout(B) == BroadcastBandedBlockBandedLayout{typeof(*)}()
            @test BandedBlockBandedMatrix(B) == B == copyto!(BandedBlockBandedMatrix(B), B) == 2*B.args[2]
            @test MemoryLayout(B') isa LazyBandedMatrices.LazyBandedBlockBandedLayout
            @test BandedBlockBandedMatrix(B') == B'
            @test MemoryLayout(Symmetric(B)) isa LazyBandedMatrices.LazyBandedBlockBandedLayout
            @test MemoryLayout(Hermitian(B)) isa LazyBandedMatrices.LazyBandedBlockBandedLayout

            C = BroadcastMatrix(*, 2, im*A)
            @test MemoryLayout(C') isa LazyBandedMatrices.LazyBandedBlockBandedLayout
            @test MemoryLayout(transpose(C)) isa LazyBandedMatrices.LazyBandedBlockBandedLayout

            E = BroadcastMatrix(*, A, 2)
            @test MemoryLayout(E) == BroadcastBandedBlockBandedLayout{typeof(*)}()

            D = Diagonal(PseudoBlockArray(randn(6),1:3))
            @test MemoryLayout(BroadcastMatrix(*, A, D)) isa BroadcastBandedBlockBandedLayout{typeof(*)}
            @test MemoryLayout(BroadcastMatrix(*, D, A)) isa BroadcastBandedBlockBandedLayout{typeof(*)}

            F = BroadcastMatrix(*, Ones(axes(A,1)), A)
            @test blockbandwidths(F) == (1,1)
            @test subblockbandwidths(F) == (1,1)
            @test F == A
        end
    end



    @testset "Apply block indexing" begin
        B = BandedBlockBandedMatrix(randn(6,6),1:3,1:3,(1,1),(1,1))
        A = ApplyArray(+, B, B)
        @test exp.(view(A,Block.(1:3),Block.(1:3))) == exp.(A)
        @test exp.(view(A,Block.(1:3),2)) == exp.(A)[Block.(1:3),2]
        @test exp.(view(A,2,Block.(1:3))) == exp.(A)[2,Block.(1:3)]
    end

    @testset "Blockbandwidths" begin
        @test blockbandwidths(unitblocks(Diagonal(1:5))) == (0,0)
    end
end

end # module