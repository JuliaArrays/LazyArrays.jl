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

@testset "lazymul/ldiv tests" begin
    A = randn(5,5)
    b = randn(5)
    @test MyMatrix(A)*b ≈ apply(*,MyMatrix(A),b) ≈ A*b
    @test MyMatrix(A)*MyMatrix(A) ≈ apply(*,MyMatrix(A),MyMatrix(A)) ≈ apply(*,MyMatrix(A),A) ≈ A^2
    @test MyMatrix(A)*MyMatrix(A)*MyMatrix(A) ≈ apply(*,MyMatrix(A),MyMatrix(A),MyMatrix(A)) ≈ A^3

    @test MyMatrix(A)\b ≈ apply(\,MyMatrix(A),b) ≈ copyto!(similar(b),Ldiv(A,copy(b))) ≈ A\b
    @test eltype(applied(\,MyMatrix(A),b)) == eltype(apply(\,MyMatrix(A),b)) == eltype(MyMatrix(A)\b) == Float64
end