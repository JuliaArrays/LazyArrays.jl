using LazyArrays, LinearAlgebra
import LazyArrays: @lazymul, @lazyldiv, materialize!

# used to test general matrix backends
struct MyMatrix{T} <: AbstractMatrix{T}
    A::Matrix{T}
end

MyMatrix{T}(::UndefInitializer, n::Int, m::Int) where T = MyMatrix{T}(Array{T}(undef, n, m))
MyMatrix(A::AbstractMatrix{T}) where T = MyMatrix{T}(Matrix{T}(A))
Base.convert(::Type{MyMatrix{T}}, A::MyMatrix{T}) where T = A
Base.convert(::Type{MyMatrix{T}}, A::MyMatrix) where T = MyMatrix(convert(AbstractArray{T}, A.A))
Base.convert(::Type{MyMatrix}, A::MyMatrix)= A
Base.convert(::Type{AbstractArray{T}}, A::MyMatrix) where T = MyMatrix(convert(AbstractArray{T}, A.A))
Base.convert(::Type{AbstractMatrix{T}}, A::MyMatrix) where T = MyMatrix(convert(AbstractArray{T}, A.A))
Base.convert(::Type{MyMatrix{T}}, A::AbstractArray{T}) where T = MyMatrix{T}(A)
Base.convert(::Type{MyMatrix{T}}, A::AbstractArray) where T = MyMatrix{T}(convert(AbstractArray{T}, A))
Base.convert(::Type{MyMatrix}, A::AbstractArray{T}) where T = MyMatrix{T}(A)
Base.getindex(A::MyMatrix, kj...) = A.A[kj...]
Base.getindex(A::MyMatrix, ::Colon, j::AbstractVector) = MyMatrix(A.A[:,j])
Base.setindex!(A::MyMatrix, v, kj...) = setindex!(A.A, v, kj...)
Base.size(A::MyMatrix) = size(A.A)
Base.similar(::Type{MyMatrix{T}}, m::Int, n::Int) where T = MyMatrix{T}(undef, m, n)
Base.similar(::MyMatrix{T}, m::Int, n::Int) where T = MyMatrix{T}(undef, m, n)
Base.similar(::MyMatrix, ::Type{T}, m::Int, n::Int) where T = MyMatrix{T}(undef, m, n)
LinearAlgebra.factorize(A::MyMatrix) = factorize(A.A)


@lazymul MyMatrix
@lazyldiv MyMatrix

struct MyLazyArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end


Base.size(A::MyLazyArray) = size(A.data)
Base.getindex(A::MyLazyArray, j::Int...) = A.data[j...]
LazyArrays.MemoryLayout(::Type{<:MyLazyArray}) = LazyLayout()
LinearAlgebra.factorize(A::MyLazyArray) = factorize(A.data)

@testset "lazymul/ldiv tests" begin
    @testset "*/" begin
        A = randn(5,5)
        B = randn(5,5)
        x = randn(5)
        @test MyMatrix(A)*x ≈ apply(*,MyMatrix(A),x) ≈ A*x
        @test MyMatrix(A)*MyMatrix(A) ≈ apply(*,MyMatrix(A),MyMatrix(A)) ≈ apply(*,MyMatrix(A),A) ≈ A^2
        @test MyMatrix(A)*MyMatrix(A)*MyMatrix(A) ≈ apply(*,MyMatrix(A),MyMatrix(A),MyMatrix(A)) ≈ A^3

        @test MyMatrix(A)\x ≈ apply(\,MyMatrix(A),x) ≈ copyto!(similar(x),Ldiv(A,copy(x))) ≈ A\x
        @test eltype(applied(\,MyMatrix(A),x)) == eltype(apply(\,MyMatrix(A),x)) == eltype(MyMatrix(A)\x) == Float64

        @test MyMatrix(A)\MyMatrix(B) ≈ MyMatrix(A)\B ≈ apply(\,MyMatrix(A),B) ≈ copyto!(similar(B),Ldiv(A,copy(B))) ≈ A\B
        @test eltype(applied(\,MyMatrix(A),B)) == eltype(apply(\,MyMatrix(A),B)) == eltype(MyMatrix(A)\B) == Float64

        @test MyMatrix(A) * ApplyArray(exp,B) ≈ apply(*, MyMatrix(A),ApplyArray(exp,B)) ≈ A*exp(B)
        @test ApplyArray(exp,A) * MyMatrix(B)  ≈ apply(*, ApplyArray(exp,A), MyMatrix(B)) ≈ exp(A)*B
        @test ApplyArray(exp,A) * ApplyArray(exp,B) ≈ apply(*, ApplyArray(exp,A),ApplyArray(exp,B)) ≈ exp(A)*exp(B)

        @test MyMatrix(A) \ x == apply(\, MyMatrix(A), x)
        @test ldiv!(MyMatrix(A), copy(x)) == materialize!(Ldiv(MyMatrix(A), copy(x)))
        @test MyMatrix(A) \ x ≈ ldiv!(MyMatrix(A), copy(x)) ≈ A\x
        @test MyMatrix(A) \ B == apply(\, MyMatrix(A), B)
        @test ldiv!(MyMatrix(A), copy(B)) == materialize!(Ldiv(MyMatrix(A), copy(B)))
        @test MyMatrix(A) \ B ≈ MyMatrix(A) \ MyMatrix(B) ≈ ldiv!(MyMatrix(A), copy(B)) ≈  A\B
        @test_broken ldiv!(MyMatrix(A), MyMatrix(copy(B))) ≈ A\B

        C = randn(5,3)
        @test all(MyMatrix(C)\x .=== apply(\,MyMatrix(C),x))
        @test MyMatrix(C)\x ≈ C\x
        @test all(MyMatrix(C)\B .=== apply(\,MyMatrix(C),B))
        @test MyMatrix(C)\B ≈ C\B

        @test_throws DimensionMismatch apply(\,MyMatrix(C),randn(4))
        @test_throws DimensionMismatch apply(\,MyMatrix(C),randn(4,3))
    end
    @testset "Lazy" begin
        A = MyLazyArray(randn(2,2))
        B = MyLazyArray(randn(2,2))
        x = MyLazyArray(randn(2))

        @test apply(*,A,x) isa ApplyVector
        @test apply(*,A,Array(x)) isa ApplyVector
        @test apply(*,Array(A),x) isa ApplyVector
        @test apply(*,A,x) == apply(*,Array(A),x) == apply(*,A,Array(x))  == Array(A)*Array(x)

        @test apply(*,A,B) isa ApplyMatrix
        @test apply(*,A,Array(B)) isa ApplyMatrix
        @test apply(*,Array(A),B) isa ApplyMatrix
        @test apply(*,A,B) == apply(*,Array(A),B) == apply(*,A,Array(B))  == Array(A)*Array(B)

        @test apply(\,A,x) isa ApplyVector
        @test apply(\,A,Array(x)) isa ApplyVector
        @test apply(\,Array(A),x) isa ApplyVector
        @test apply(\,A,x) == apply(\,Array(A),x) == apply(\,A,Array(x))  == Array(A)\Array(x)

        @test apply(\,A,B) isa ApplyMatrix
        @test apply(\,A,Array(B)) isa ApplyMatrix
        @test apply(\,Array(A),B) isa ApplyMatrix
        @test apply(\,A,B) ≈ apply(\,Array(A),B) ≈ apply(\,A,Array(B))  ≈ Array(A)\Array(B)
    end
end