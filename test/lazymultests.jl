module LazyMulTests

using LazyArrays, ArrayLayouts, LinearAlgebra, FillArrays
import LazyArrays: materialize!, MemoryLayout, triangulardata, LazyLayout, UnknownLayout, LazyMatrix, simplifiable
using Test

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
Base.getindex(A::MyMatrix, k::Integer, j::Integer) = A.A[k, j]
Base.getindex(A::MyMatrix, k::Integer, ::Colon) = A.A[k,:]
Base.getindex(A::MyMatrix, ::Colon, j::Integer) = A.A[:,j]
Base.getindex(A::MyMatrix, ::Colon, j::AbstractVector) = MyMatrix(A.A[:,j])
Base.setindex!(A::MyMatrix, v, kj...) = setindex!(A.A, v, kj...)
Base.size(A::MyMatrix) = size(A.A)
Base.copy(A::MyMatrix) = MyMatrix(copy(A.A))
Base.similar(::Type{MyMatrix{T}}, m::Int, n::Int) where T = MyMatrix{T}(undef, m, n)
Base.similar(::MyMatrix{T}, m::Int, n::Int) where T = MyMatrix{T}(undef, m, n)
Base.similar(::MyMatrix, ::Type{T}, m::Int, n::Int) where T = MyMatrix{T}(undef, m, n)
LinearAlgebra.factorize(A::MyMatrix) = factorize(A.A)

struct MyLazyArray{T,N} <: LazyArray{T,N}
    data::Array{T,N}
end

Base.size(A::MyLazyArray) = size(A.data)
Base.getindex(A::MyLazyArray, j::Int...) = A.data[j...]
Base.copy(M::MyLazyArray) = MyLazyArray(copy(M.data))
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
        @test ldiv!(MyMatrix(A), MyMatrix(copy(B))) ≈ A\B

        C = randn(5,3)
        @test all(MyMatrix(C)\x .=== apply(\,MyMatrix(C),x))
        @test MyMatrix(C)\x ≈ C\x
        @test all(MyMatrix(C)\B .=== apply(\,MyMatrix(C),B))
        @test MyMatrix(C)\B ≈ C\B

        @test_throws DimensionMismatch apply(\,MyMatrix(C),randn(4))
        @test_throws DimensionMismatch apply(\,MyMatrix(C),randn(4,3))
    end

    @testset "/" begin
        A = randn(5,5)
        B = randn(5,5)

        Ap = applied(/, A, B)
        @test axes(Ap) == (axes(Ap,1), axes(Ap,2))
        @test size(Ap) == size(A)

        @test MyMatrix(A) / B == apply(/, MyMatrix(A), B)
        @test MyMatrix(A) / B ≈ MyMatrix(A) / MyMatrix(B) ≈ A / MyMatrix(B) ≈  A/B

        @test ApplyArray(/, MyMatrix(A), B) \ A ≈ ApplyArray(/, A, B) \ A ≈ ApplyArray(/, A, B) \ MyMatrix(A) ≈ B
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
        @test Q * B ≈ Q * B.A
        @test B * Q ≈ B.A * Q
        @test apply(*, B) ≈ B
        @test Matrix(apply(*, Q)) ≈ Matrix(Q)
        @test apply(*, 1, Q) ≈ apply(*, Q, 1) ≈ Matrix(Q)
        @test apply(*, Q, Q') ≈ apply(*, Q', Q) ≈ Matrix(I, 3, 3)
        @test apply(*, Q, Q', B) ≈ apply(*, B, Q, Q') ≈ B.A

        Q = qr(randn(5,3)).Q
        @test Q * B ≈ Q*B.A
        @test B * Q' ≈ B.A * Q'
        @test apply(*, B) ≈ B
        @test Matrix(apply(*, Q)) ≈ Matrix(Q)
        @test apply(*, 1, Q) ≈ apply(*, Q, 1) ≈ Q * 1
        @test apply(*, Q, Q') ≈ apply(*, Q', Q) ≈ Matrix(I, 5, 5)
        
        B = MyMatrix(randn(5,5))
        @test Q * B ≈ Q * B.A
        @test B * Q ≈ B.A * Q
        @test apply(*, B) ≈ B
        @test Matrix(apply(*, Q)) ≈ Matrix(Q)
        @test apply(*, 1, Q) ≈ apply(*, Q, 1) ≈ 1*Q
        @test apply(*, Q, Q') ≈ apply(*, Q', Q) ≈ Matrix(I, 5, 5)
        @test apply(*, Q, Q', B) ≈ apply(*, B, Q, Q') ≈ B.A
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

        @test @inferred(ArrayLayouts.ldiv(MyMatrix(A), M)) ≈ A\ B * b
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

        @test MemoryLayout(Tridiagonal(c, b, c)) isa TridiagonalLayout{LazyLayout,LazyLayout,LazyLayout}
        @test MemoryLayout(SymTridiagonal(b, c)) isa SymTridiagonalLayout{LazyLayout,LazyLayout}
        @test MemoryLayout(Bidiagonal(b, c, :U)) isa BidiagonalLayout{LazyLayout,LazyLayout}

        @test LazyArrays.tridiagonallayout(UnknownLayout(), UnknownLayout(), LazyLayout()) isa TridiagonalLayout{LazyLayout,LazyLayout,LazyLayout}
        @test LazyArrays.tridiagonallayout(UnknownLayout(), LazyLayout(), UnknownLayout()) isa TridiagonalLayout{LazyLayout,LazyLayout,LazyLayout}
        @test LazyArrays.tridiagonallayout(LazyLayout(), UnknownLayout(), UnknownLayout()) isa TridiagonalLayout{LazyLayout,LazyLayout,LazyLayout}
        @test LazyArrays.tridiagonallayout(UnknownLayout(), LazyLayout(), LazyLayout()) isa TridiagonalLayout{LazyLayout,LazyLayout,LazyLayout}
        @test LazyArrays.tridiagonallayout(LazyLayout(), UnknownLayout(), LazyLayout()) isa TridiagonalLayout{LazyLayout,LazyLayout,LazyLayout}
        @test LazyArrays.tridiagonallayout(LazyLayout(), LazyLayout(), UnknownLayout()) isa TridiagonalLayout{LazyLayout,LazyLayout,LazyLayout}

        @test LazyArrays.symtridiagonallayout(UnknownLayout(), LazyLayout()) isa SymTridiagonalLayout{LazyLayout,LazyLayout}
        @test LazyArrays.symtridiagonallayout(LazyLayout(), UnknownLayout()) isa SymTridiagonalLayout{LazyLayout,LazyLayout}

        @test LazyArrays.bidiagonallayout(UnknownLayout(), LazyLayout()) isa BidiagonalLayout{LazyLayout,LazyLayout}
        @test LazyArrays.bidiagonallayout(LazyLayout(), UnknownLayout()) isa BidiagonalLayout{LazyLayout,LazyLayout}
    end

    @testset "Nested" begin
        a = MyLazyArray(randn(5))
        @test a .\ rand(5) .* Zeros(5) ≡ Zeros(5)
        @test broadcast(*, Zeros(5), Base.broadcasted(\, a, rand(5))) ≡ Zeros(5)
    end

    @testset "inv" begin
        A = randn(5,5)
        B = randn(5,5)
        M = ApplyArray(*, A, B)
        b = randn(5)
        @test M \ MyLazyArray(b) ≈ M \ b
    end

    @testset "Diagonal Fill" begin
        b = randn(5)
        B = randn(5,5)
        @test Eye(5) * MyLazyArray(b) == b
        @test MyLazyArray(B) * Eye(5) == B

        @test simplifiable(*, Eye(5), MyLazyArray(B)) isa Val{true}
        @test simplifiable(*, Eye(5), Eye(5)) isa Val{true}
        @test simplifiable(*, MyLazyArray(B), Eye(5)) isa Val{true}

        @test simplifiable(*, Zeros(5), Zeros(5)') isa Val{true}
        @test simplifiable(*, Zeros(5), randn(5)') isa Val{true}
        @test simplifiable(*, randn(5), Zeros(5)') isa Val{true}
        @test simplifiable(*, randn(5), Eye(5)) isa Val{true}
        @test simplifiable(*, Eye(5), Zeros(5)) isa Val{true}
        @test simplifiable(*, Zeros(5,5), Eye(5)) isa Val{true}
    end

    @testset "LazyBroadcast" begin
        a = MyLazyArray(randn(5))
        b = a .^ 2
        @test BroadcastArray(view(b,1:3)) == Vector(a)[1:3] .^2
    end

    @testset "Apply*Broadcast" begin
        A = randn(5,5)
        B = randn(5,5)

        @test ApplyArray(*, A, B) * BroadcastArray(*, A, B) ≈ (A*B) * (A .* B)
        @test BroadcastArray(*, A, B) * ApplyArray(*, A, B) ≈ (A .* B) * (A*B)

        @test ApplyArray(*, A, B) \ BroadcastArray(*, A, B) ≈ (A*B) \ (A .* B)
        @test BroadcastArray(*, A, B) \ ApplyArray(*, A, B) ≈ (A .* B) \ (A * B)
        @test BroadcastArray(*, A, B) \ BroadcastArray(*, A, B) ≈ (A .* B) \ (A .* B)
    end

    @testset "CartesianIndex view" begin
        A = randn(5,5)
        B = randn(5,5)
        M = ApplyArray(*, A, B)
        @test layout_getindex(M,[CartesianIndex(1,2),CartesianIndex(3,3)]) == M[[CartesianIndex(1,2),CartesianIndex(3,3)]] == [M[1,2], M[3,3]]
    end

    @testset "(A\\B)*C" begin
        A = randn(5,5)
        B = randn(5,5)
        C = randn(5,5)
        L = ApplyArray(\,A,B)
        @test L*C ≈ L*MyMatrix(C) ≈ A \ (B*C)
    end

    @testset "Dual" begin
        A = MyLazyArray(randn(5,5))
        x = MyLazyArray(randn(5))
        @test x'A*x ≈ x.data'A.data*x.data
        @test x'A'*x ≈ x.data'A.data'*x.data
        @test x'A*A ≈ x.data'A.data*A.data
        @test x'A*A*x ≈ x.data'A.data*A.data*x.data

        @test simplifiable(*, x'A, x) == Val(true)
    end

    @testset "Zeros" begin
        A = MyLazyArray(randn(5,5))
        @test A * Zeros(5) ≡ mul(A, Zeros(5)) ≡ Zeros(5)
        @test A * Zeros(5,2) ≡ mul(A, Zeros(5,2)) ≡ Zeros(5,2)

        @test Zeros(5)' * A ≡ Zeros(5)'
        @test transpose(Zeros(5)) * A ≡ transpose(Zeros(5))

        D = Diagonal(1:5)
        y = MyLazyArray(randn(5))

        @test Zeros(5)' * D * y == transpose(Zeros(5)) * D * y == 0.0
        @test y' * D * Zeros(5) == transpose(y) * D * Zeros(5) == 0.0
        @test Zeros(5)' * Diagonal(y) ≡ Zeros(5)'
        @test transpose(Zeros(5)) * Diagonal(y) ≡ transpose(Zeros(5))
        @test Zeros(5)' * Diagonal(y) * y == 0.0
        @test transpose(Zeros(5)) * Diagonal(y) * y == 0.0
        @test y' * Diagonal(y) * Zeros(5) == 0.0
        @test transpose(y) * Diagonal(y) * Zeros(5) == 0.0
        @test Zeros(5)' * Diagonal(y) * Zeros(5) == 0.0
        @test transpose(Zeros(5)) * Diagonal(y) * Zeros(5) == 0.0
    end

    @testset "Vector LazyMul" begin
        A = ApplyMatrix(*, [[1 2], [3 4]], permutedims([[1, 2], [3,4]]))
        Ã = *(A.args...)
        @test A == [A[k,j] for k=1:2, j=1:2] == Ã
        @test_broken A[2,:] == Ã[2,:] # TODO: need to rethink arguments permuting
    end

    @testset "Tri Lazy" begin
        A = MyMatrix(randn(5,5))
        b = Vcat(1:3,Zeros(2))
        @test UpperTriangular(A)*b isa Vector
        @test UpperTriangular(A)*b ≈ UpperTriangular(A.A)*b
        @test (UpperTriangular(A)UpperTriangular(A)) * b isa Vector
        @test (UpperTriangular(A)UpperTriangular(A)) * b ≈UpperTriangular(A.A)^2 * b
    end

    @testset "DualLayout{ApplyLayout{typeof(*)}} translayout" begin
        @test ApplyArray(*, (1:2)', [1 2; 3 4]) * [1 2; 3 4] ≈ [37,54]'
        @test ApplyArray(*, (1:2)', [1 2; 3 4]) * [1 2; 3 4] isa Adjoint
    end
end

end # module
