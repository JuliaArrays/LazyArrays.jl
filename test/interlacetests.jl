using LazyArrays, ArrayLayouts, Test
import LazyArrays: InterlaceLayout, arguments

@testset "Interlace" begin
    n = 10
    a = 1:n
    b = n+1:2n
    A = Vcat(a', b')
    v = vec(A)
    @test MemoryLayout(v) isa InterlaceLayout
    @test arguments(v) == (a,b)
    @test ArrayLayouts._copyto!(zeros(2n), v) == vec(Matrix(A))

    v = view(A, 1:2n)
    MemoryLayout(v)
end