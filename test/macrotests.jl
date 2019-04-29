module MacroTests

using Test, LazyArrays, MacroTools

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
        @test lazy isa Union{Broadcast.Broadcasted, LazyArrays.Applied}

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

end  # module
