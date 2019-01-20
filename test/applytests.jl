using LazyArrays, Test
import LazyArrays: materialize, broadcasted, DefaultApplyStyle, Applied

@testset "Applied" begin
    @test applied(exp,1) isa Applied{DefaultApplyStyle}

    @test materialize(applied(randn)) isa Float64

    @test materialize(applied(*, 1)) == 1

    @test materialize(applied(exp, 1)) === exp(1)
    @test materialize(applied(exp, broadcasted(+, 1, 2))) ===
          materialize(applied(exp, applied(+, 1, 2))) === exp(3)
end
