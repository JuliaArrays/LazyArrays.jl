using LazyArrays, FillArrays, Test
import LazyArrays: materialize, broadcasted, DefaultApplyStyle, Applied,
            ApplyArray, ApplyMatrix, ApplyVector

@testset "Applied" begin
    @test applied(exp,1) isa Applied{DefaultApplyStyle}

    @test apply(randn) isa Float64

    @test materialize(applied(*, 1)) == apply(*,1) == 1

    @test apply(exp, 1) === exp(1)
    @test apply(exp, broadcasted(+, 1, 2)) ===
    apply(exp, applied(+, 1, 2)) === exp(3)
end

@testset "ApplyArray" begin
    A = randn(2,2)
    M = ApplyMatrix(exp, A)
    @test eltype(M) == Float64
    @test M == exp(A)

    b = randn(2)
    c = ApplyVector(*, ApplyMatrix(exp, A), b);

    @test axes(c) == (Base.OneTo(2),)

    @test c[1] == c[1,1]
    @test exp(A)*b ≈ c

    @test ApplyArray(+,[1,2],[3,4]) == ApplyVector(+,[1,2],[3,4]) ==
            ApplyArray(+,[1,2],[3,4])

    @test LazyArrays.rowsupport(Diagonal(1:10),3) == 3:3
    @test ApplyArray(*, Ones(100_000_000,100_000_000), Diagonal(1:100_000_000))[1:10,1:10] == 
            ones(10,10)*Diagonal(1:10)
end
