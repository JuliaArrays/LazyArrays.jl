using LazyArrays, Test
using LazyArrays.SetOperations
import LazyArrays.SetOperations: ∪, ∩, unioned, intersected, setdiffed

@testset "Set operations" begin
    @test SetOperations.Style(Int) == SetOperations.VectorStyle()
    @test SetOperations.Style(Set) == SetOperations.SetStyle()
    @test 1 ∈ unioned(1,2.0)
    @test 2 ∈ unioned(1,2.0)
    @test 3 ∉ unioned(1,2.0)
    @test issubset([1,2],  unioned(1,2.0))

    @test 1 ∪ 2.0 == [1.0,2.0]
    @test [1,2] ∪ Set([2,4]) ==  Set([2,4]) ∪ [1,2] == SetOperations.union(Set([2,4]) , [1,2]) ==
        SetOperations.union([1,2], Set([2,4])) == Set([1,2,4])
    @test Base.union([1,2], Set([2,4])) == Base.:∪([1,2], Set([2,4])) ==
                [1,2,4]

    @test 1 ∈ intersected([1,2], Set([1.0]))
    @test 2 ∉ intersected([1,2], Set([1.0]))
    @test 1 ∉ setdiffed([1,2], Set([1.0]))
    @test 2 ∈ setdiffed([1,2], Set([1.0]))


    @test 1 ∈ intersected([1,2], Set([1.0]), 1f0)
end
