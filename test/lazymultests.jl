using LazyArrays, ArrayLayouts, LinearAlgebra, FillArrays
import LazyArrays: materialize!, MemoryLayout, triangulardata, LazyLayout, UnknownLayout, LazyMatrix

# used to test general matrix backends
struct MyMatrix{T} <: LazyMatrix{T}
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
Base.getindex(A::MyMatrix, ::Colon, j::Integer) = A.A[:,j]
Base.getindex(A::MyMatrix, ::Colon, j::AbstractVector) = MyMatrix(A.A[:,j])
Base.setindex!(A::MyMatrix, v, kj...) = setindex!(A.A, v, kj...)
Base.size(A::MyMatrix) = size(A.A)
Base.similar(::Type{MyMatrix{T}}, m::Int, n::Int) where T = MyMatrix{T}(undef, m, n)
Base.similar(::MyMatrix{T}, m::Int, n::Int) where T = MyMatrix{T}(undef, m, n)
Base.similar(::MyMatrix, ::Type{T}, m::Int, n::Int) where T = MyMatrix{T}(undef, m, n)
LinearAlgebra.factorize(A::MyMatrix) = factorize(A.A)

struct MyLazyArray{T,N} <: LazyArray{T,N}
    data::Array{T,N}
end

Base.size(A::MyLazyArray) = size(A.data)
Base.getindex(A::MyLazyArray, j::Int...) = A.data[j...]
LazyArrays.MemoryLayout(::Type{<:MyLazyArray}) = LazyLayout()
LinearAlgebra.factorize(A::MyLazyArray) = factorize(A.data)

@testset "lazymul/ldiv tests" begin
    @testset "*" begin
        A = randn(5,5)
        B = randn(5,5)
        x = randn(5)
        @test MyMatrix(A)*x ≈ apply(*,MyMatrix(A),x) ≈ A*x
        @test MemoryLayout(MyMatrix(A)) isa LazyLayout
        @test all(MyMatrix(A)*MyMatrix(A) .=== apply(*,MyMatrix(A),MyMatrix(A)))
        @test all(MyMatrix(A)*A .=== apply(*,MyMatrix(A),A))
        @test all(A*MyMatrix(A) .=== apply(*,A,MyMatrix(A)))
        @test MyMatrix(A)*MyMatrix(A) ≈ MyMatrix(A)*A ≈ A*MyMatrix(A) ≈ A^2

        @test MyMatrix(A)*MyMatrix(A)*MyMatrix(A) ≈ apply(*,MyMatrix(A),MyMatrix(A),MyMatrix(A)) ≈ A^3

        @test all(UpperTriangular(A) * MyMatrix(A) .=== apply(*,UpperTriangular(A), MyMatrix(A)))
        @test all(MyMatrix(A) * UpperTriangular(A) .=== apply(*, MyMatrix(A),UpperTriangular(A)))
        @test all(Diagonal(A) * MyMatrix(A) .=== apply(*,Diagonal(A), MyMatrix(A)))
        @test all(MyMatrix(A) * Diagonal(A) .=== apply(*, MyMatrix(A),Diagonal(A)))

        @test all(MyMatrix(A)' * x .=== apply(*,MyMatrix(A)',x))

        @test all(MyMatrix(A)' * MyMatrix(A)' .=== apply(*,MyMatrix(A)', MyMatrix(A)'))
        @test all(MyMatrix(A)' * A' .=== apply(*,MyMatrix(A)', A'))
        @test all(A' * MyMatrix(A)' .=== apply(*,MyMatrix(A)', MyMatrix(A)'))
        @test all(MyMatrix(A)' * MyMatrix(A) .=== apply(*,MyMatrix(A)', MyMatrix(A)))
        @test all(MyMatrix(A)' * A .=== apply(*,MyMatrix(A)', A))
        @test all(MyMatrix(A) * MyMatrix(A)' .=== apply(*,MyMatrix(A), MyMatrix(A)'))
        @test all(A * MyMatrix(A)' .=== apply(*,A, MyMatrix(A)'))

        @test all(UpperTriangular(A) * MyMatrix(A) .=== apply(*,UpperTriangular(A), MyMatrix(A)))
        @test all(MyMatrix(A) * UpperTriangular(A) .=== apply(*, MyMatrix(A),UpperTriangular(A)))

        @test all(Diagonal(A) * MyMatrix(A)' .=== apply(*,Diagonal(A), MyMatrix(A)'))
        @test all(MyMatrix(A)' * Diagonal(A) .=== apply(*,MyMatrix(A)',Diagonal(A)))
        @test all(UpperTriangular(A) * MyMatrix(A)' .=== apply(*,UpperTriangular(A), MyMatrix(A)'))
        @test all(MyMatrix(A)' * UpperTriangular(A) .=== apply(*,MyMatrix(A)',UpperTriangular(A)))

        @test ApplyArray(\, MyMatrix(A), x)[1,1] ≈ (A\x)[1]
        @test MyMatrix(A)\x ≈ apply(\,MyMatrix(A),x) ≈ copyto!(similar(x),Ldiv(A,copy(x))) ≈ A\x
        @test eltype(applied(\,MyMatrix(A),x)) == eltype(apply(\,MyMatrix(A),x)) == eltype(MyMatrix(A)\x) == Float64

        @test MyMatrix(A)\MyMatrix(B) ≈ MyMatrix(A)\B ≈ apply(\,MyMatrix(A),B) ≈ copyto!(similar(B),Ldiv(A,copy(B))) ≈ A\B
        @test eltype(applied(\,MyMatrix(A),B)) == eltype(apply(\,MyMatrix(A),B)) == eltype(MyMatrix(A)\B) == Float64

        @test MyMatrix(A) * ApplyArray(exp,B) ≈ apply(*, MyMatrix(A),ApplyArray(exp,B)) ≈ A*exp(B)
        @test ApplyArray(exp,A) * MyMatrix(B)  ≈ apply(*, ApplyArray(exp,A), MyMatrix(B)) ≈ exp(A)*B
        @test ApplyArray(exp,A) * ApplyArray(exp,B) ≈ apply(*, ApplyArray(exp,A),ApplyArray(exp,B)) ≈ exp(A)*exp(B)

        @test MyMatrix(A) * BroadcastArray(exp,B) ≈ apply(*, MyMatrix(A),BroadcastArray(exp,B)) ≈ A*exp.(B)
        @test BroadcastArray(exp,A) * MyMatrix(B)  ≈ apply(*, BroadcastArray(exp,A), MyMatrix(B)) ≈ exp.(A)*B
        @test BroadcastArray(exp,A) * BroadcastArray(exp,B) ≈ apply(*, BroadcastArray(exp,A),BroadcastArray(exp,B)) ≈ exp.(A)*exp.(B)
    end

    @testset "\\" begin
        A = randn(5,5)
        B = randn(5,5)
        x = randn(5)
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
        @test apply(*,A,x) ≈ apply(*,Array(A),x) ≈ apply(*,A,Array(x))  ≈ Array(A)*Array(x)

        @test apply(*,A,B) isa ApplyMatrix
        @test apply(*,A,Array(B)) isa ApplyMatrix
        @test apply(*,Array(A),B) isa ApplyMatrix
        @test apply(*,A,B) ≈ apply(*,Array(A),B) ≈ apply(*,A,Array(B))  ≈ Array(A)*Array(B)

        @test apply(\,A,x) isa ApplyVector
        @test apply(\,A,Array(x)) isa ApplyVector
        @test apply(\,Array(A),x) isa ApplyVector
        @test apply(\,A,x) ≈ apply(\,Array(A),x) ≈ apply(\,A,Array(x)) ≈ Array(A)\Array(x)

        @test apply(\,A,B) isa ApplyMatrix
        @test apply(\,A,Array(B)) isa ApplyMatrix
        @test apply(\,Array(A),B) isa ApplyMatrix
        @test apply(\,A,B) ≈ apply(\,Array(A),B) ≈ apply(\,A,Array(B)) ≈ Array(A)\Array(B)

        Ap = applied(*,A,x)
        @test copyto!(similar(Ap), Ap) == A*x
        @test copyto!(similar(Ap,BigFloat), Ap) ≈ A*x

        @test MemoryLayout(typeof(Diagonal(x))) isa DiagonalLayout{LazyLayout}
        @test MemoryLayout(typeof(Diagonal(ApplyArray(+,x,x)))) isa DiagonalLayout{LazyLayout}
        @test MemoryLayout(typeof(Diagonal(1:6))) isa DiagonalLayout{UnknownLayout}

        @test MemoryLayout(typeof(A')) isa LazyLayout
        @test MemoryLayout(typeof(transpose(A))) isa LazyLayout
        @test MemoryLayout(typeof(view(A,1:2,1:2))) isa LazyLayout
        @test MemoryLayout(typeof(reshape(A,4))) isa LazyLayout
    end

    @testset "QR" begin
        B = MyMatrix(randn(3,3))
        Q = qr(randn(3,3)).Q
        @test Q * B ≈ Q*B.A
    end

    @testset "ambiguities" begin
        A = randn(5,5)
        b = MyLazyArray(randn(5))
        c = randn(5)
        c̃ = complex.(c)
        @test A*b isa ApplyVector{Float64,typeof(*)}
        @test UpperTriangular(A)*b isa ApplyVector{Float64,typeof(*)}
        @test A*b ≈ A*Vector(b)
        @test UpperTriangular(A)*b ≈ UpperTriangular(A)*Vector(b)
        @test c'b ≈ c̃'b ≈ c'Vector(b)
        @test transpose(c)b ≈ transpose(c̃)b ≈ transpose(c)Vector(b)
    end

    @testset "InvMatrix" begin
        A = randn(5,5)
        B = randn(5,5)
        b = MyLazyArray(randn(5))
        M = ApplyArray(*, B, b)
        @test InvMatrix(A) * b ≈ A \ b
        @test InvMatrix(A) * M ≈ A \ B * b

        @test ArrayLayouts.ldiv(MyMatrix(A), M) ≈ A\ B * b
    end

    @testset "Tri/Diagonal" begin 
        b = MyLazyArray(randn(5))
        c = MyLazyArray(randn(4))
        d = MyLazyArray(randn(3))
        @test copy(Diagonal(b)) == Diagonal(copy(b))
        @test map(copy, Diagonal(b))  == Diagonal(copy(b))
        @test inv(Diagonal(b)) == inv(Diagonal(b.data))
        @test inv(Diagonal(b)) isa Diagonal{Float64,<:BroadcastVector}
        @test copy(Tridiagonal(c, b, c)) == Tridiagonal(copy(c), copy(b), copy(c))
        @test copy(Tridiagonal(c, b, c, d)) == Tridiagonal(copy(c), copy(b), copy(c), copy(d))
        @test copy(Tridiagonal(c, b, c, d)).du2 == d
        @test map(copy, Tridiagonal(c, b, c)) == Tridiagonal(copy(c), copy(b), copy(c))
        @test map(copy, Tridiagonal(c, b, c, d)) == Tridiagonal(copy(c), copy(b), copy(c), copy(d))
        @test map(copy, Tridiagonal(c, b, c, d)).du2 == d
    end

    @testset "Nested" begin
        a = MyLazyArray(randn(5))
        @test a .\ rand(5) .* Zeros(5) ≡ Zeros(5)
        @test broadcast(*, Zeros(5), Base.broadcasted(\, a, rand(5))) ≡ Zeros(5)
    end

    @testset "Inv of Mul" begin
        A = randn(5,5)
        B = randn(5,5)
        M = ApplyArray(*, A, B)
        @test inv(M) ≈ inv(A*B)
        @test_throws DimensionMismatch inv(ApplyArray(*,randn(5,6), rand(6,5)))
    end
end