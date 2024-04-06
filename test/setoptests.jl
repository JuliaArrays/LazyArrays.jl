module SetOpTests

using LazyArrays, Test
import LazyArrays: ApplyStyle, VectorSetStyle, SetStyle

@testset "Set operations" begin
    @test ApplyStyle(union,Int) == ApplyStyle(intersect,Int) == ApplyStyle(setdiff,Int) == VectorSetStyle()
    @test ApplyStyle(union,Set) == ApplyStyle(intersect,Set) == ApplyStyle(setdiff,Set) == SetStyle()
    @test 1 ∈ applied(union,1,2.0)
    @test 2 ∈ applied(union,1,2.0)
    @test 3 ∉ applied(union,1,2.0)
    @test issubset([1,2],  applied(union,1,2.0))

    @test eltype(apply(∪, 1, 2.0)) == Float64
    @test  apply(∪, 1, 2.0) == [1.0,2.0]
    @test apply(∪, [1,2], Set([2,4])) ==  Set([2,4]) ∪ [1,2] == apply(union, Set([2,4]) , [1,2]) ==
        apply(union, [1,2], Set([2,4])) == Set([1,2,4])
    @test Base.union([1,2], Set([2,4])) == Base.:∪([1,2], Set([2,4])) ==
                [1,2,4]

    @test 1 ∈ applied(intersect, [1,2], Set([1.0]))
    @test 2 ∉ applied(intersect, [1,2], Set([1.0]))
    @test 1 ∉ applied(setdiff, [1,2], Set([1.0]))
    @test 2 ∈ applied(setdiff, [1,2], Set([1.0]))

    @test 1 ∈ applied(intersect, [1,2], Set([1.0]), 1f0)
end # testset

end # module
