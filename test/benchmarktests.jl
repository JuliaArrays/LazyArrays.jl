# These tests use BenchmarkTools, and are unreliable in an
# uncontrolled environment (like Travis)

using LazyArrays, BenchmarkTools

@testset "Applied" begin
    @test @belapsed(apply(exp, $x)) ≤ 2(@belapsed exp($x))
end

@testset "concat" begin
    A = Vcat(Vector(1:10), Vector(1:20))
    b = Array{Int}(undef, 30)
    @test @belapsed(copyto!($b,$A)) < @belapsed(vcat($A.args...))

    A = Vcat(1:10, 1:20)
    b = Array{Int}(undef, 30)
    @test @belapsed(copyto!($b,$A)) < @belapsed(vcat($A.args...))

    A = Vcat(randn(2,10), randn(4,10))
    b = Array{Float64}(undef, 6,10)
    @test @belapsed(copyto!($b,$A)) < @belapsed(vcat($A.args...))

    A = Hcat(1:10, 2:11)
    b = Array{Int}(undef, 10, 2)
    @test @belapsed(copyto!($b,$A)) < @belapsed(hcat($A.args...))
end
