module MacroTests

using Test, LazyArrays, MacroTools
using Base.Broadcast: Broadcasted

A = randn(6, 6)
B = BroadcastArray(+, A, 2)
C = randn(6, 6)

expressions_block = quote
    exp.(A)
    @. exp(A)
    # exp(A)
    A .+ 2
    @. A + 2
    A + B
    @. A + B
    A * B + C
    # A * B .+ C
    A * (B + C)
    # A * (B .+ C)
    # 2 .* (A * B) .+ 3 .* C
end
testparams = [
    ("$(rmlines(ex))", ex) for ex in expressions_block.args if ex isa Expr
]

@testset "@~" begin
    @testset "$label" for (label, ex) in testparams
        desired = @eval $ex
        lazy = @eval @~ $ex
        @test lazy isa Union{Broadcasted, Applied}

        @testset ".= @~ $label" begin
            actual = zero(desired)
            actual .= lazy
            @test actual == desired
        end

        @testset "materialize(@~ $label)" begin
            @test materialize(lazy) == desired
        end

        @testset "LazyArray(@~ $label)" begin
            actual = LazyArray(lazy) :: LazyArray
            @test actual == desired
        end

        @testset "materialize(LazyArray(@~ $label))" begin
            @test materialize(LazyArray(lazy)) == desired
        end

        @testset ".= LazyArray(@~ $label)" begin
            actual = zero(desired)
            actual .= LazyArray(lazy)
            @test actual == desired
        end
    end
end

@testset "@~ laziness" begin
    A = ones(1, 1)
    x = [1]

    bc = @~ exp.(A * x)
    @test bc.args isa Tuple{Applied}

    bc = @~ exp.(A * x .+ 1)
    @test bc.args isa Tuple{Broadcasted}
    @test bc.args[1].args isa Tuple{Applied, Int}
end

end  # module
