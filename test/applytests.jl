module ApplyTests

using LazyArrays, FillArrays, ArrayLayouts, Test
import LazyArrays: materialize, broadcasted, DefaultApplyStyle, Applied, arguments,
            ApplyArray, ApplyMatrix, ApplyVector, LazyArrayApplyStyle, ApplyLayout, call
import ArrayLayouts: StridedLayout
using LinearAlgebra

@testset "Applying" begin
    @testset "Applied" begin
        @test applied(exp,1) isa Applied{DefaultApplyStyle}
        @test apply(randn) isa Float64
        @test materialize(applied(*, 1)) == apply(*,1) == 1
        @test apply(exp, 1) === exp(1)
        @test apply(exp, broadcasted(+, 1, 2)) === apply(exp, applied(+, 1, 2)) === exp(3)
    end

    @testset "ApplyArray" begin
        A = randn(2,2)
        M = ApplyMatrix(exp, A)
        @test eltype(M) == eltype(Applied(M)) == Float64
        @test M == exp(A)
        @test ndims(M) == 2
        @test axes(M) == (Base.OneTo(2), Base.OneTo(2))

        b = randn(2)
        c = ApplyVector(*, ApplyMatrix(exp, A), b)

        @test axes(c) == (Base.OneTo(2),)

        @test c[1] == c[1,1]
        @test exp(A)*b ≈ c

        @test ApplyArray(+,[1,2],[3,4]) == ApplyVector(+,[1,2],[3,4]) == ApplyArray(+,[1,2],[3,4])

        M = ApplyArray(*, Ones(100_000_000,100_000_000), Diagonal(1:100_000_000))
        @test M[1,1] === 1.0
        @test M[1:10,1:10] == ones(10,10)*Diagonal(1:10)

        M = copy(Applied{LazyArrayApplyStyle}(exp, (A,)))
        @test M isa ApplyArray
        @test M == exp(A)
    end

    @testset "copy (#85)" begin
        v = ApplyVector(vec, ones(Int, 2, 2))
        vc = copy(float.(v))
        ve = convert(Vector, vc)
        @test eltype(ve) == Float64

        A = ApplyArray(exp, randn(5,5))
        @test copy(A) ≡ map(copy,A) ≡ A
    end

    @testset "copyto!" begin
        a = ApplyArray(+,[1,2],[3,4])
        b = ApplyArray(+,[3,4],[5,6])
        c = ApplyArray(+,[3.0,4.0],[5.0,6.0])
        @test copyto!(b, a) == a == b
        @test copyto!(c, a) == a == c
        @test copyto!(similar(a), a) == a
        @test_throws Base.CanonicalIndexError copyto!(c, Array(a))
    end

    @testset "vec" begin
        A = zeros(2,2)
        v = applied(vec, A)
        w = vec(A)
        @test size(v) == size(w) == (4,)
        @test axes(v) == axes(w) == (Base.OneTo(4),)
        @test ndims(v) == ndims(w) == 1
        @test eltype(v) == eltype(w) == Float64
    end

    @testset "view" begin
        a = ApplyArray(+,[1,2,3],[3,4,5])
        v = view(a,1:2)
        @test MemoryLayout(typeof(v)) isa ApplyLayout{typeof(+)}
        @test call(v) == call(a) == +
        @test Array(v) == a[1:2] == Array(a)[1:2]
        @test v == ApplyArray(v) == ApplyArray{Float64}(v) == ApplyVector(v) == ApplyVector{Float64}(v)

        A = ApplyArray(+,[1 2; 3 4],[3 4; 5 6])
        V = view(A, 1:2, :)
        @test V == ApplyArray(V) == ApplyArray{Float64}(V) == ApplyMatrix(V) == ApplyMatrix{Float64}(V)
    end

    @testset "rot180" begin
        A = randn(3,2)
        R = ApplyArray(rot180, A)
        @test eltype(R) == Float64
        @test size(R) == size(A)
        @test MemoryLayout(R) isa StridedLayout
        @test strides(R) == (-1,-3)
        @test pointer(R) == pointer(view(A,3,2))
        @test R == rot180(A) == copyto!(similar(A),R)

        R = ApplyArray(rotl90, A)
        @test eltype(R) == Float64
        @test size(R) == (2,3)
        @test R == rotl90(A)

        R = ApplyArray(rotr90, A)
        @test eltype(R) == Float64
        @test size(R) == (2,3)
        @test R == rotr90(A)

        B = randn(2,3)
        R = ApplyArray(rot180, ApplyArray(*, A, B))
        @test MemoryLayout(R) isa ApplyLayout{typeof(*)}
        @test arguments(R) == (rot180(A), rot180(B))
        @test R ≈ rot180(A*B)
    end

    @testset "scalar * Matrix" begin
        A = randn(5,5)
        M = ApplyArray(*,1.0,A)
        @test M ≈ A
        for k = axes(A,1), j = axes(A,2)
            @test M[k,j] ≈ A[k,j]
        end
        @test colsupport(M,1) == 1:5
    end
end # testset

end # module
