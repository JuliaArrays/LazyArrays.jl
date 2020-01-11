using LazyArrays, FillArrays, Test
import LazyArrays: CachedArray, CachedMatrix, CachedVector

@testset "Cache" begin
    @testset "basics" begin
        A = 1:10
        C = cache(A)
        @test size(C) == (10,)
        @test axes(C) == (Base.OneTo(10),)
        @test all(Vector(C) .=== Vector(A))
        @test cache(C) isa CachedArray{Int,1,Vector{Int},UnitRange{Int}}
        C2 = cache(C)
        @test C2.data !== C.data
        @test C[:] == C[Base.Slice(Base.OneTo(10))] == C

        @test cache(A)[2,1] == 2
        @test_throws BoundsError cache(A)[2,2]

        A = reshape(1:10^2, 10,10)
        C = cache(A)
        @test size(C) == (10,10)
        @test axes(C) == (Base.OneTo(10),Base.OneTo(10))
        @test all(Array(C) .=== Array(A))
        @test cache(C) isa CachedArray{Int,2,Matrix{Int},typeof(A)}
        C2 = cache(C)
        @test C2.data !== C.data
        @test C[1:2,1:2] == [1 11; 2 12]

        A = reshape(1:10^3, 10,10,10)
        C = cache(A)
        @test size(C) == (10,10,10)
        @test axes(C) == (Base.OneTo(10),Base.OneTo(10),Base.OneTo(10))
        @test all(Array(C) .=== Array(A))

        A = reshape(1:10^3, 10,10,10)
        C = cache(A)
        LazyArrays.resizedata!(C,5,5,5)
        LazyArrays.resizedata!(C,8,8,8)
        @test all(C.data .=== Array(A)[1:8,1:8,1:8])

        @test C[1:3,1,:] == A[1:3,1,:]
    end

    @testset "Matrix cache" begin
        A = collect(1:5)
        C = cache(A)
        @test C isa Vector{Int}
        C[1] = 2
        @test A[1] ≠ 2

        A = cache(Matrix(reshape(1:6,2,3)))
        C = cache(A)
        @test C isa Matrix{Int}
        C[1,1] = 2
        @test A[1,1] ≠ 2
        C[1] = 3
        @test A[1,1] ≠ 3
    end

    @testset "setindex!" begin
        A = (1:5)
        C = cache(A)
        C[1] = 2
        @test C[1] == 2

        A = cache(reshape(1:6,2,3))
        C = cache(A)
        C[1,1] = 2
        @test A[1,1] ≠ 2
        @test C[1,1] == 2
        C[1] = 3
        @test C[1,1] == 3

        @test_throws BoundsError C[3,1]
        @test_throws BoundsError C[7]
    end

    @testset "colsupport past size" begin
        C = cache(Zeros(5,5)); C[5,1];
        @test colsupport(C,1) == Base.OneTo(5)
        @test colsupport(C,3) == 1:0
        @test rowsupport(C,1) == Base.OneTo(1)
    end

    @testset "broadcast" begin
        x = CachedArray([1,2,3],1:8);
        y = 1:8;
        f = (x,y) -> cos(x*y)
        @test f.(x,y) isa CachedArray
        @test @inferred(broadcast(f,x,y)) == f.(Vector(x), Vector(y))

        @test (x + y) isa CachedArray
        @test (x + y).array isa AbstractRange
        @test (x + y) == Vector(x) + Vector(y)

        z = CachedArray([1,4],Zeros{Int}(8));
        @test (x .+ z) isa CachedArray
        @test (x + z) isa CachedArray
        @test Vector( x .+ z) == Vector( x + z) == Vector(x) + Vector(z)

        # Lazy mixed with Static treats as Lazy
        s = SVector(1,2,3,4,5,6,7,8)
        @test f.(x , s) isa CachedArray
        @test f.(x , s) == f.(Vector(x), Vector(s))
    end

    @testset "padded CachedVector getindex" begin
        v = CachedArray([1,2,3],Zeros{Int}(1000))
        @test v[3:100] == [3; zeros(Int,97)]
        @test v[3:100] isa CachedArray{Int,1,Vector{Int},Zeros{Int,1,Tuple{Base.OneTo{Int}}}}
        @test v[4:end] isa CachedArray{Int,1,Vector{Int},Zeros{Int,1,Tuple{Base.OneTo{Int}}}}
        @test all(iszero,v[4:end])
        @test isempty(v[4:0])
        v = CachedArray([1,2,3],Fill{Int}(1,1000))
        @test v[3:100] == [3; ones(97)]
        @test norm(v) == norm(Array(v)) == norm(v,2)
        @test norm(v,1) == norm(Array(v),1)
        @test norm(v,Inf) == norm(Array(v),Inf)
        @test norm(v,3) ≈ norm(Array(v),3)
    end

    @testset "ambiguity broadcast" begin
        c = cache(1:100)
        v = Vcat([1,2,3],0:96)
        z = Zeros(100)
        @test v .+ c == c .+ v == Array(c) + Array(v)
        @test z .+ c == c .+ z == Array(c)
    end

    @testset "Fill" begin
        c = cache(Fill(1,10))
        @test c[2:3] isa LazyArrays.CachedVector{Int,Vector{Int},Fill{Int,1,Tuple{Base.OneTo{Int}}}}
        @test c[[2,4,6]] isa LazyArrays.CachedVector{Int,Vector{Int},Fill{Int,1,Tuple{Base.OneTo{Int}}}}
    end

    @testset "linalg" begin
        c = cache(Fill(3,3,3))
        @test fill(2,1,3) * c == fill(18,1,3)
        @test ApplyMatrix(exp,fill(3,3,3)) * c == exp(fill(3,3,3)) * fill(3,3,3)
        @test BroadcastMatrix(exp,fill(3,3,3)) * c == exp.(fill(3,3,3)) * fill(3,3,3)
        @test fill(2,3)' * c == fill(18,1,3)
        @test fill(2,3,1)' * c == fill(18,1,3)
    end

    @testset "broadcast" begin
        c = cache(Fill(3.0,10)); c[1] = 2;
        @test exp.(c) isa typeof(c)
        @test exp.(c) == exp.(Vector(c))
        @test c .+ 1 isa typeof(c)
        @test c .+ 1 == Vector(c) .+ 1
        @test 1 .+ c isa typeof(c)
        @test 1 .+ c == 1 .+ Vector(c)
        @test c .+ Ref(1) isa typeof(c)
        @test Ref(1) .+ c isa typeof(c)
    end

    @testset "*" begin
        A = randn(2,2)
        B = ApplyMatrix(exp,A)
        C = BroadcastMatrix(exp,A)
        x = cache(Fill(3,2))
        @test A*x == A*Vector(x)
        @test B*x == Matrix(B)*Vector(x)
        @test C*x == Matrix(C)*Vector(x)
        @test A'x == Matrix(A)'Vector(x)
        @test B'x == Matrix(B)'Vector(x)
        @test C'x == Matrix(C)'Vector(x)
    end
end