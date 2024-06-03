module MacroTests

using Test, LazyArrays, MacroTools
using Base.Broadcast: Broadcasted

A = randn(6, 6)
B = BroadcastArray(+, A, 2)
C = randn(6, 6)

expressions_block = quote
    exp.(A)
    @. exp(A)
    exp(A)
    A .+ 2
    @. A + 2
    A + B
    @. A + B
    A * B + C
    A * B .+ C
    A * (B + C)
    # A * (B .+ C)
    2 .* (A * B) .+ 3 .* C
    exp.(A * C)  # https://github.com/JuliaArrays/LazyArrays.jl/issues/54
    (A * A) .+ (A * C)
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
            @test actual ≈ desired
        end

        @testset "materialize(@~ $label)" begin
            @test materialize(lazy) ≈ desired
        end

        @testset "LazyArray(@~ $label)" begin
            actual = LazyArray(lazy) :: LazyArray
            @test actual ≈ desired
        end

        @testset "materialize(LazyArray(@~ $label))" begin
            @test_skip materialize(LazyArray(lazy)) == desired  # should work
            @test materialize(LazyArray(lazy)) ≈ desired
        end

        @testset ".= LazyArray(@~ $label)" begin
            actual = zero(desired)
            actual .= LazyArray(lazy)
            @test_skip actual == desired  # should work
            @test actual ≈ desired
        end
    end
end

struct CustomProperty end
Base.getproperty(::CustomProperty, property::Symbol) = property
Base.getproperty(::CustomProperty, property) = property

complex_number = 1 + 2im
custom_property = CustomProperty()

expressions_block = quote
    complex_number.im # https://github.com/JuliaArrays/LazyArrays.jl/pull/69
    custom_property."property"
end
testparams = [
    ("$(rmlines(ex))", ex) for ex in expressions_block.args if ex isa Expr
]

@testset "@~ non-lazy" begin
    @testset "$label" for (label, ex) in testparams
        desired = @eval $ex
        actual = @eval @~ $ex
        @test actual === desired
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

@testset "@~ and `@views`" begin
    # https://github.com/JuliaArrays/LazyArrays.jl/issues/144
    A = randn(6, 6)
    @test copy(@views @~ exp.(A[1:2, 1:2])) == exp.(A[1:2, 1:2])
end

@testset "error capturing" begin
    @test_throws "ArgumentError: @~ is capturing more than one expression, try capturing \"u - v\" with brackets" @macroexpand @~ u - v, 2
end

end  # module
