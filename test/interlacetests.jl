using LazyArrays, Test

@testset "Interlace" begin
    @test_throws ArgumentError Interlace(1:5, 10:15)
    @test_throws ArgumentError Interlace(1:5, 10:12)
    @test eltype(Interlace(1:5, 10:13)) == Int
    @test Interlace(1:5, 10:13) == interlace(1:5,10:13) == [1,10,2,11,3,12,4,13,5]
    @test Interlace(1:5, 10:14) == interlace(1:5,10:14) == [1,10,2,11,3,12,4,13,5,14]
end