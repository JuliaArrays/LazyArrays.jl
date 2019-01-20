using LazyArrays, Test
import LazyArrays: materialize, broadcasted, DefaultApplyStyle, Applied,
            ApplyArray, ApplyMatrix, ApplyVector

@testset "Applied" begin
    @test applied(exp,1) isa Applied{DefaultApplyStyle}

    @test materialize(applied(randn)) isa Float64

    @test materialize(applied(*, 1)) == 1

    @test materialize(applied(exp, 1)) === exp(1)
    @test materialize(applied(exp, broadcasted(+, 1, 2))) ===
          materialize(applied(exp, applied(+, 1, 2))) === exp(3)
end

@testset "ApplyArray" begin
    A = randn(2,2)
    M = ApplyMatrix(exp, A)
    @test eltype(M) == Float64
    @test M == exp(A)

    b = randn(2)
    c = MulVector(ApplyMatrix(exp, A), b)

    @test axes(c) == (Base.OneTo(2),)

    @test c[1] == c[1,1]
    @test exp(A)*b == c

    @test ApplyArray(+,[1,2],[3,4]) == ApplyVector(+,[1,2],[3,4]) ==
            ApplyArray(+,[1,2],[3,4])
end
