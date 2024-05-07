module LazyArraysBlockArraysTest
using LazyArrays, ArrayLayouts, BlockArrays, FillArrays, StaticArrays, Test
using LazyArrays: LazyArrayStyle, PaddedLayout, PaddedColumns, PaddedRows, paddeddata
using BlockArrays: blockcolsupport, blockrowsupport

@testset "Lazy BlockArrays" begin
    @testset "LazyBlock" begin
        @test Block(5) in BroadcastVector(Block, [1,3,5])
        @test Base.broadcasted(LazyArrayStyle{1}(), Block, 1:5) ≡ Block.(1:5)
        @test Base.broadcasted(LazyArrayStyle{1}(), Int, Block.(1:5)) ≡ 1:5
        @test Base.broadcasted(LazyArrayStyle{0}(), Int, Block(1)) ≡ 1
    end

    @testset "LazyBlockArray Triangle Recurrences" begin
        N = 1000
        n = mortar(BroadcastArray(Fill,Base.OneTo(N),Base.OneTo(N)))
        k = mortar(BroadcastArray(Base.OneTo,Base.OneTo(N)))

        @test view(n, Block(5)) ≡ Fill(5,5)
        @test view(k,Block(5)) ≡ Base.OneTo(5)
        a = b = c = 0.0
        # for some reason the following was causing major slowdown. I think it
        # went pass a limit to Base.Broadcast.flatten which caused `bc.f` to have a strange type.
        # bc = Base.Broadcast.instantiate(Base.broadcasted(/, Base.broadcasted(*, k, Base.broadcasted(-, Base.broadcasted(-, k, n), a)), Base.broadcasted(+, 2k, b+c-1)))

        bc = Base.Broadcast.instantiate(Base.broadcasted((k,n,a,b,c) -> k * (k-n-a) / (2k+(b+c-1)), k, n, a, b, c))
        @test axes(n,1) ≡ axes(k,1) ≡ axes(bc)[1] ≡ blockedrange(Base.OneTo(N))
        u = (k .* (k .- n .- a) ./ (2k .+ (b+c-1)))
        @test u == (Vector(k) .* (Vector(k) .- Vector(n) .- a) ./ (2Vector(k) .+ (b+c-1)))
        @test copyto!(u, bc) == (k .* (k .- n .- a) ./ (2k .+ (b+c-1)))
        @test @allocated(copyto!(u, bc)) ≤ 1000
        # not clear why allocatinos so high: all allocations are coming from checking
        # axes

        u = PseudoBlockArray{Float64}(undef, collect(1:N))
        @test copyto!(u, bc) == (k .* (k .- n .- a) ./ (2k .+ (b+c-1)))
        @test @allocated(copyto!(u, bc)) ≤ 1000
    end

    @testset "padded" begin
        c = Vcat(randn(3), Zeros(7))
        b = PseudoBlockVector(c, 1:4)
        @test MemoryLayout(b) isa PaddedColumns
        @test b[Block.(2:3)] isa PseudoBlockVector{Float64,<:ApplyArray}
        @test MemoryLayout(b[Block.(2:3)]) isa PaddedColumns
        @test b[Block.(2:3)] == b[2:6] 
        
        c = PseudoBlockVector(Vcat(1, Zeros(5)), 1:3)
        @test paddeddata(c) == [1]
        @test paddeddata(c) isa PseudoBlockVector
        @test blockcolsupport(c) == Block.(1:1)
        C = PseudoBlockArray(Vcat(randn(2,3), Zeros(4,3)), 1:3, [1,2])
        @test blockcolsupport(C) == Block.(1:2)
        @test blockrowsupport(C) == Block.(1:2)
        
        @test C[Block.(1:2),1:3] == C[Block.(1:2),Block.(1:2)] == C[1:3,Block.(1:2)] == C[1:3,1:3]
        
        H = PseudoBlockArray(Hcat(1, Zeros(1,5)), [1], 1:3)
        @test MemoryLayout(H) isa PaddedRows
        @test paddeddata(H) == Ones(1,1)
        
        b = PseudoBlockArray(cache(Zeros(55)),1:10);
        b[10] = 5;
        @test MemoryLayout(b) isa PaddedColumns{DenseColumnMajor}
        @test paddeddata(b) isa PseudoBlockVector
        @test paddeddata(b) == [zeros(9); 5]
    end

    @testset "Lazy block" begin
        b = PseudoBlockVector(randn(5),[2,3])
        c = BroadcastVector(exp,1:5)
        @test c .* b isa BroadcastVector
        @test b .* c isa BroadcastVector
        @test (c .* b)[Block(1)] == c[1:2] .* b[Block(1)]

        b = PseudoBlockVector(randn(5),[2,3])
        a = ApplyArray(+, b, b)

        @test exp.(view(a,Block.(1:2))) == exp.(a)
    end

    @testset "PaddedArray" begin
        p = PaddedArray(1:5, (blockedrange(1:4),))
        @test paddeddata(p) == [1:5; 0]
        @test blocksize(paddeddata(p),1) == 3
    end

    @testset "blockedrange" begin
        b = blockedrange(SVector{2}([1,2]))
        @test b .+ b == 2:2:6
    end
end
end