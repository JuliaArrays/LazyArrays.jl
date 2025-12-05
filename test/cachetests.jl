module CacheTests

using LazyArrays, FillArrays, LinearAlgebra, ArrayLayouts, SparseArrays, Test
using StaticArrays
import LazyArrays: CachedArray, CachedMatrix, CachedVector, PaddedLayout, CachedLayout, resizedata!, zero!,
                    CachedAbstractArray, CachedAbstractVector, CachedAbstractMatrix, AbstractCachedArray, AbstractCachedMatrix,
                    PaddedColumns, cacheddata, LazyArrayStyle, maybe_cacheddata, Accumulate, CachedArrayStyle, GenericCachedLayout,
                    AccumulateAbstractVector
import Base.Broadcast: BroadcastStyle

using ..InfiniteArrays
using .InfiniteArrays: OneToInf
using Infinities

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
        @test C[:] == C[Base.Slice(Base.OneTo(10))] == C[1:10] == C[collect(1:10)] == C
        @test C[CartesianIndex(2)] == 2

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

        @test C[1,:] == C[1,1:10] == C[1,collect(1:10)] == A[1,1:10]
        @test C[:,1] == C[1:10,1] == C[collect(1:10),1] == A[1:10,1]
        @test C[1:10,1:10] == C[collect(1:10),collect(1:10)] == C[:,:] == C[1:10,:] ==
                C[collect(1:10),:] == C[:,1:10] == C[:,collect(1:10)] == A

        @test C[1:5] == C[1:5,1]
        @test C[1:11] == [C[:,1]; C[1,2]]

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

    @testset "convert" begin
        C = cache(1:5);
        C[1] = 3;
        @test convert(AbstractVector{Int}, C) ≡ convert(AbstractArray{Int}, C) ≡ C
        @test convert(AbstractVector{Float64}, C) isa CachedVector
        @test convert(AbstractArray{Float64}, C) isa CachedVector
        @test convert(AbstractVector{Float64}, C) == convert(AbstractArray{Float64}, C) == C
    end

    @testset "Matrix cache" begin
        A = collect(1:5)
        C = cache(A)
        @test C isa Vector{Int}
        C[1] = 2
        @test A[1] ≠ 2
        @test resizedata!(C,6) === C

        A = cache(Matrix(reshape(1:6,2,3)));
        C = cache(A);
        @test C isa Matrix{Int}
        C[1,1] = 2
        @test A[1,1] ≠ 2
        C[1] = 3
        @test A[1,1] ≠ 3
        @test resizedata!(C,2,3) === C
    end

    @testset "setindex!" begin
        A = (1:5)
        C = cache(A)
        C[1] = 2
        @test C[1] == 2

        A = cache(reshape(1:6,2,3))
        C = cache(A)
        C[1,1] = 2
        @test A[1,1] ≠ 2
        @test C[1,1] == 2
        C[1] = 3
        @test C[1,1] == 3

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

        @test Zeros(8) .* x ≡ x .* Zeros(8) ≡ Zeros(8)

        @testset "Padded" begin
            z = CachedArray([1,4],Zeros{Int}(8));
            @test (x .+ z) isa CachedArray
            @test (x + z) isa CachedArray
            @test Vector( x .+ z) == Vector( x + z) == Vector(x) + Vector(z)
        end

        @testset "Lazy mixed with Static treats as Lazy" begin
            s = SVector(1,2,3,4,5,6,7,8)
            @test f.(x , s) isa CachedArray
            @test f.(x , s) == f.(Vector(x), Vector(s))
        end

        @testset "sub-matrix" begin
            A = cache(Zeros(8,8));
            b = BroadcastVector(exp,randn(8))
            @test view(A, :, 1) .+ x == x .+ view(A, :, 1) == A[:,1] .+ x
            @test view(A, :, 1) .+ b == b .+ view(A, :, 1) == A[:,1] .+ b

            A = cache(Zeros(8,8));
            view(A, :, 2)[3] = 4
            @test A[3,2] == 4
        end
    end

    @testset "padded CachedVector getindex" begin
        v = CachedArray([1,2,3],Zeros{Int}(1000))
        @test v[3:100] == [3; zeros(Int,97)]
        @test v[3:100] isa CachedArray{Int,1,Vector{Int},Zeros{Int,1,Tuple{Base.OneTo{Int}}}}
        @test v[4:end] isa CachedArray{Int,1,Vector{Int},Zeros{Int,1,Tuple{Base.OneTo{Int}}}}
        @test MemoryLayout(v) isa PaddedColumns
        @test all(iszero,v[4:end])
        @test isempty(v[4:0])
        @test norm(v) == norm(Array(v)) == norm(v,2)
        @test norm(v,1) == norm(Array(v),1)
        @test norm(v,Inf) == norm(Array(v),Inf)
        @test norm(v,3) == norm(Array(v),3)
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
        @test c .* z ≡ z .* c ≡ z
    end

    @testset "Fill" begin
        c = cache(Fill(1,10))
        @test c[2:3] isa LazyArrays.CachedVector{Int,Vector{Int},Fill{Int,1,Tuple{Base.OneTo{Int}}}}
        @test c[[2,4,6]] isa LazyArrays.CachedVector{Int,Vector{Int},Fill{Int,1,Tuple{Base.OneTo{Int}}}}

        F = Fill(2, ℵ₀)
        C = cumsum(cache(F))
        @test axes(C) == (InfiniteArrays.OneToInf(),)
    end

    @testset "linalg" begin
        c = cache(Fill(3,3,3))
        @test fill(2,1,3) * c == fill(18,1,3)
        @test ApplyMatrix(exp,fill(3,3,3)) * c ≈ exp(fill(3,3,3)) * fill(3,3,3)
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
        D = Diagonal(randn(2))
        x = cache(Fill(3,2))
        @test A*x ≈ A*Vector(x)
        @test B*x ≈ Matrix(B)*Vector(x)
        @test C*x ≈ Matrix(C)*Vector(x)
        @test D*x ≈ Matrix(D)*Vector(x)
        @test A'x ≈ Matrix(A)'Vector(x)
        @test B'x ≈ Matrix(B)'Vector(x)
        @test C'x ≈ Matrix(C)'Vector(x)

        @testset "padded" begin
            z = cache(Zeros(2));
            @test MemoryLayout(z) isa PaddedColumns
            @test A*z ≈ A*Vector(z)
            @test B*z ≈ Matrix(B)*Vector(z)
            @test C*z ≈ Matrix(C)*Vector(z)
            @test D*z ≈ Matrix(D)*Vector(z)
            @test A'z ≈ Matrix(A)'Vector(z)
            @test B'z ≈ Matrix(B)'Vector(z)
            @test C'z ≈ Matrix(C)'Vector(z)

            p = Vcat([1,2],Zeros(3)) + cache(Zeros(5))
            @test MemoryLayout(p) isa PaddedColumns
            @test p isa CachedVector{Float64,Vector{Float64},<:Zeros}
            @test p == [1; 2; zeros(3)]
        end
    end

    @testset "copyto!" begin
        a = CachedArray([1,2,3], Zeros{Int}(8));
        b = CachedArray(Int[], Zeros{Int}(8));
        c = CachedArray(Float64[], Zeros{Float64}(8));
        @test copyto!(b, a) == a == b
        @test copyto!(c, a) == a == c

        @test copyto!(a, Zeros{Int}(8)) == zeros(8)
        a = CachedArray([1,2,3], Zeros{Int}(8));
        copyto!(view(a,3:8), Zeros{Int}(6))
        @test a == [1; 2; zeros(6)]

        a = CachedArray([3,missing], Zeros{Union{Int,Missing}}(4));
        b = CachedArray(Union{Int,Missing}[], Zeros{Union{Int,Missing}}(4));
        @test all(copyto!(b, a) .=== a .=== b)

        a = CachedArray([1,2,3], Zeros{Int}(8));
        copyto!(view(a,1:2), [4,5])
        @test a[1:2] == [4,5]
        a = CachedArray(Int[], Zeros{Int}(8));
        copyto!(view(a,1:2), view(Vcat([4,5],Zeros(6)),1:2))
    end

    @testset "fill!/lmul!/rmul!" begin
        a = CachedArray(Zeros{Float64}(100_000_000));
        a[3] = 4;
        fill!(a, 0.0);
        @test a.datasize[1] == 3
        @test a[1:3] == zeros(3)
        a[3] = 4;
        rmul!(a, 2.0);
        @test a.datasize[1] == 3
        @test a[1:3] == [0,0,8]
        lmul!(2.0, a);
        @test a.datasize[1] == 3
        @test a[1:3] == [0,0,16]

        @test_throws ArgumentError fill!(a, 1.0)
        @test_throws ArgumentError rmul!(a, Inf)
        @test_throws ArgumentError lmul!(Inf, a)

        a = CachedArray(Ones{Float64}(100_000_000));
        lmul!(5, view(a, 1:3))
        @test a[1:5] == [fill(5,3); 1; 1]
        rmul!(view(a, 4:5), 6)
        @test a[1:7] == [fill(5,3); fill(6,2); fill(1,2)]
    end

    @testset "Padded broadcast" begin
        a = CachedArray([1,2,3], Zeros{Int}(8));
        r = a .- a;
        @test MemoryLayout(r) isa PaddedColumns
        @test r.datasize[1] == 3
        @test r == Vector(a) - Vector(a)

        a = CachedArray([1,2,3], Zeros{Int}(8));
        b = Fill(2,8);
        r = a .- b;
        @test MemoryLayout(r) isa CachedLayout{DenseColumnMajor,FillLayout}
        @test r.datasize[1] == 3
        @test r == Vector(a) - Vector(b)

        a = CachedArray([1,2,3], Zeros{Int}(8));
        b = Fill(2,8);
        r = b .- a;
        @test MemoryLayout(r) isa CachedLayout{DenseColumnMajor,FillLayout}
        @test r.datasize[1] == 3
        @test r == Vector(b) - Vector(a)

        a = CachedArray([1,2,3], Zeros{Int}(8));
        b = CachedArray([1,2],Fill(2,8));
        r = a .- b;
        @test MemoryLayout(r) isa CachedLayout{DenseColumnMajor,FillLayout}
        @test r.datasize[1] == 3
        @test r == Vector(a) - Vector(b)

        b = CachedArray([1,2,3], Zeros{Int}(8));
        a = CachedArray([1,2],Fill(2,8));
        r = a .- b;
        @test MemoryLayout(r) isa CachedLayout{DenseColumnMajor,FillLayout}
        @test r.datasize[1] == 3
        @test r == Vector(a) - Vector(b)

        a = CachedArray([1,2,3], Zeros{Int}(8));
        @test_throws DimensionMismatch a .+ CachedArray([1,2],Fill(2,6))
    end

    @testset "BigFloat" begin
        a = cache(Zeros{BigFloat}(10));
        resizedata!(a,3);
        resizedata!(a,10);
        @test a == Zeros{BigFloat}(10)
    end

    @testset "zero!" begin
        a = cache(Zeros(10));
        @test zero!(a) ≡ a
        @test a.datasize == (0,)
    end

    @testset "minimum/maximum" begin
        c = cache(Fill(2,4));
        c[1] = 1;
        @test maximum(c) == 2
        @test minimum(c) == 1
        @test sum(c) == 7
        c[1] = 3;
        @test maximum(c) == 3
        @test minimum(c) == 2
        c[1:4] .= 1;
        @test maximum(c) == 1
        c[1:4] .= 3;
        @test minimum(c) == 3
    end

    @testset "permutedims" begin
        a = cache(Zeros(10));
        a[3] = 2
        a.data[4:end] .= NaN # fill with noise
        @test permutedims(a) isa CachedMatrix
        @test permutedims(a).datasize == (1,3)
        @test permutedims(a)[4] == permutedims(a)[1,4] == 0.0
        @test permutedims(a) == [0 0 2 zeros(1,7)]

        A = cache(Zeros(5,6));
        A[2,3] = 2;
        @test permutedims(A) isa CachedMatrix
        @test permutedims(A).datasize == (3,2)
    end

    @testset "CachedAbstract" begin
        A = CachedAbstractArray(1:5);
        A[2] = 3
        @test -A isa CachedAbstractVector
        @test A .+ 1 isa CachedAbstractVector
        @test A .+ Ref(1) isa CachedAbstractVector
        @test 1 .+ A isa CachedAbstractVector
        @test Ref(1) .+ A isa CachedAbstractVector
        @test A .+ A isa CachedAbstractVector

        @test CachedAbstractVector(1:5) isa CachedAbstractVector
        @test CachedAbstractMatrix(reshape(1:6,2,3)) isa CachedAbstractMatrix
    end

    @testset "getindex and data structure" begin
        A = Matrix(1.0I[1:100,1:100])
        A[2,1] = 3.
        Ac = cache(sparse(A))
        @test Ac isa AbstractCachedArray
        @test Ac isa AbstractCachedMatrix
        @test issparse(Ac[1:10,1:10])
        @test issparse(Ac[1:3,1:5])
        @test issparse(Ac[Vector(1:3),Vector(1:5)])
        @test issparse(Ac[1,Vector(1:5)])
        @test issparse(Ac[Vector(1:5),6])
        @test issparse(Ac[1,1:end])
        @test issparse(Ac[1,1:10])
        @test issparse(Ac[1:3,1:5])
        @test issparse(Ac[1:10,end])
        @test Ac[1,1] == 1.0
    end

    @testset "Cached triangular getindex" begin
        for Tri in (:UnitUpperTriangular, :UpperTriangular, :UnitLowerTriangular, :LowerTriangular)
            @eval begin
                A = Matrix(1.0I[1:100,1:100])
                A[2,1] = 1.1873
                A[4,3] = 8.1230
                Ac = cache($Tri(A))
                T = $Tri(Ac);
                @test T[1:10,1:20] == $Tri(A)[1:10,1:20]
                @test T[1:100,1:100] == $Tri(A)[1:100,1:100]
                @test T[Vector(1:10),Vector(1:10)] == $Tri(A)[1:10,1:10]
                @test T[1:10,1:end] == $Tri(A)[1:10,1:end]
                @test T[1,1:end] isa Vector{eltype(T)}
                @test T[1,Vector(1:7)] isa Vector{eltype(T)}
                @test T[1:5,1] isa Vector{eltype(T)}
                @test T[Vector(1:5),1] isa Vector{eltype(T)}
                @test T[:,1] == $Tri(A)[:,1]
                @test T[5,:] == $Tri(A)[5,:]
                @test T[5:10,:] == $Tri(A)[5:10,:]
                @test T[:,1:3] == $Tri(A)[:,1:3]
                @test T[:,:] == $Tri(A)[:,:]
                @test T[Vector(5:10),:] == $Tri(A)[Vector(5:10),:]
                @test T[:,Vector(1:3)] == $Tri(A)[:,Vector(1:3)]
            end
        end
    end

    @testset "Adj/Trans" begin
        v = cache(1:100)
        @test v'[1:5] == transpose(v)[1:5] == v'[1,1:5] == transpose(v)[1,1:5] == v[1:5]

        A = cache(reshape(1:110, 10, 11));
        @test A'[1,1:5] == transpose(A)[1,1:5] == A[1:5,1]
        @test  A'[1:3] == A.array'[1:3]
        @test  A'[1:11] == A.array'[1:11]
    end

    @testset "AbstractQ" begin
        Q = qr(randn(5,5)).Q
        C = cache(Q);
        @test C[1:5,1:5] == Q[1:5,1:5]
        @test length(C) == 25
    end

    @testset "fill! views" begin
        a = CachedArray(Zeros(100));
        view(a, 2:2:5) .= 2;
        @test a.datasize == (4,)
        @test a[1:5] == [0,2,0,2,0]

        a = CachedArray(Zeros(ℵ₀));
        a[4] = 2;
        view(a, 2:ℵ₀) .= 0.0;
        @test a[1:5] == zeros(5)
    end

    @testset "Issue #327" begin
        A = cache(Zeros((1:5, OneToInf())))
        B = cache(Zeros((1:5, OneToInf())))
        @test A == B
        A[2, 7] = 2.0
        @test A ≠ B
        B[2, 7] = 2.0 
        @test A == B

        A = cache(Zeros((OneToInf(), 1:7)))
        B = cache(Zeros((OneToInf(), 1:10)))
        @test A ≠ B 
        B = cache(Zeros((OneToInf(), 1:7)))
        @test A == B 
        B[2, 2] = 1.0
        @test A ≠ B 
        A[2, 2] = 1.0 
        @test A == B 

        A = cache(Zeros((OneToInf(), OneToInf())))
        B = cache(Zeros((OneToInf(), OneToInf())))
        @test A == B 
        A[5, 7] = 3.4
        @test A ≠ B 
        B[5, 7] = 3.4
        @test A == B
    end

    @testset "copyto! with CachedArrayStyle" begin
        a = Accumulate(*, 1:5);
        b = BroadcastVector(*, 2, a);
        dest = Vector{Int}(undef, 3)
        src = view(b, 1:3);
        bc = LazyArrays._broadcastarray2broadcasted(src);
        @test similar(bc, Float32) == cache(zeros(Float32, 3)) && similar(bc, Float32) isa CachedArray{Float32}
        @test a.datasize == (1,);
        @inferred LazyArrays.resize_bcargs!(bc);
        @test a.datasize == (3,)
        dest = Vector{Int}(undef, 1)
        src = view(b, 5:5);
        bc = LazyArrays._broadcastarray2broadcasted(src);
        @inferred LazyArrays.resize_bcargs!(bc);
        @test a.datasize == (5,)

        a = Accumulate(*, 1:5); # reset to test different resizing
        b = BroadcastVector(*, 2, a); 
        dest = Vector{Int}(undef, 4)
        src = view(b,2:5)
        bc = LazyArrays._broadcastarray2broadcasted(src);
        rbc = LazyArrays.resize_bcargs!(bc);
        @test Base.Broadcast.BroadcastStyle(typeof(rbc)) == Base.Broadcast.DefaultArrayStyle{1}() 
        @test rbc.f === bc.f 
        @test rbc.args == (2, a[2:5])

        a = Accumulate(*, 1:5); # reset to ensure copyto! is working as intended
        b = BroadcastVector(*, 2, a);
        dest = Vector{Int}(undef, 3);
        src = view(b,2:4);
        copyto!(dest, src)
        @test dest == [4,12,48]

        @testset "Matrix" begin
            a = view(Accumulate(*, 1:10), 1:2)
            b = rand(2, 6)
            src = a .\ b 
            dest = zeros(2, 6)
            res = Vector(a) .\ b 
            copyto!(dest, src)
            @test dest == res
        end

        @testset "Avoid StackOverflow for recursive CachedArrayStyles" begin
            @test Matrix(view((1:5)', :, 1:1) .* view(Accumulate(*, 1:5)', :, 1:1)) == [1;;] # used to StackOverflow
        end

        @testset "DualLayout{<:AbstractCachedLayout}" begin
            arg1 = view((1:100)', :, 1:10)
            arg2 = view(AccumulateAbstractVector(*, 1:100)', :, 1:10)
            bc = Base.Broadcast.Broadcasted(CachedArrayStyle{2}(), *, (arg1, arg2));
            rsz_bc = LazyArrays.resize_bcargs!(bc);
            @test rsz_bc.args[2] == view(arg2.parent.parent.data', :, 1:10)
        end
    end
                                            
    @testset "maybe_cacheddata" begin
        A = cache(1:10)
        @test maybe_cacheddata(A) === cacheddata(A)
        B = view(A, 1:5)
        @test maybe_cacheddata(B) === cacheddata(B)
        C = [1, 2, 3]
        @test maybe_cacheddata(C) === C

        v = cache(1:10)'
        @test maybe_cacheddata(v) === cacheddata(parent(v))'
        v = transpose(cache(1:10))
        @test maybe_cacheddata(v) === transpose(cacheddata(parent(v)))
    end

    @testset "Missing BroadcastStyles/MemoryLayouts/cacheddata with CachedArrayStyles" begin
        A = view(Accumulate(*, [1, 2, 3])', 1:1, 1:2)
        B = view(transpose(Accumulate(*, [1, 2im, 3])), 1:1, 1:2)
        C = Accumulate(*, [1, 2im, 3])'
        D = transpose(Accumulate(*, [1, 2im, 3]))
        E = view(Accumulate(*, [1, 2im, 3])', 1:1, 1:2)
        F = view(Accumulate(*, [1, 2, 3]), 1:2)'
        G = view(Accumulate(*, [1, 2im, 3])', 1:1, 1:2)'
        @test all(==(CachedArrayStyle{1}()), Base.BroadcastStyle.(typeof.((A, B, C, D, E, F, G))))
        @test all(==(GenericCachedLayout()), MemoryLayout.(typeof.((A, B, E, G))))
        @test all(==(DualLayout{GenericCachedLayout}()), MemoryLayout.(typeof.((C, D, F))))
        @test MemoryLayout(typeof(C)) == DualLayout{GenericCachedLayout}()
        @test MemoryLayout(typeof(D)) == DualLayout{GenericCachedLayout}()
        @test cacheddata(A) === view(cacheddata(parent(parent(A)))', 1:1, 1:1)
        @test cacheddata(B) === view(transpose(cacheddata(parent(parent(B)))), 1:1, 1:1)
        @test cacheddata(C) === cacheddata(parent(C))'
        @test cacheddata(D) === transpose(cacheddata(parent(D)))
        @test cacheddata(E) === view(cacheddata(parent(parent(E)))', 1:1, 1:1)
        @test cacheddata(F) === view(cacheddata(parent(parent(F))), 1:1)'
        @test cacheddata(G) === adjoint(view(cacheddata(parent(G)), 1:1, 1:1))
    end

    @testset "BroadcastStyle for Vcat/Hcat of CachedArrayStyles" begin
        @test BroadcastStyle(typeof(Vcat(cache(1:3), cache(4:6)))) == CachedArrayStyle{1}()
        d = Accumulate(*, 1:10)
        @test BroadcastStyle(typeof(Vcat(d, d))) == CachedArrayStyle{1}()
        @test BroadcastStyle(typeof(Vcat(d', d'))) == CachedArrayStyle{2}()
        @test BroadcastStyle(typeof(Hcat(d, d))) == CachedArrayStyle{2}()
        @test BroadcastStyle(typeof(Vcat(d', (1:10)'))) == CachedArrayStyle{2}()
        @test BroadcastStyle(typeof(Vcat((1:10)', d'))) == CachedArrayStyle{2}()
        @test BroadcastStyle(typeof(Hcat(d, (1:10)))) == CachedArrayStyle{2}()
        @test BroadcastStyle(typeof(Hcat((1:10), d))) == CachedArrayStyle{2}()
    end

    #=
    @testset "Enforce same-size arguments for cacheddata" begin
        @testset "max_datasize" begin
            @test LazyArrays._datasize(1:10) == (10,)
            x = cache(1:10)
            @test LazyArrays._datasize(x) == (0,)
            resizedata!(x, 3)
            @test LazyArrays._datasize(x) == (3,)
            @test LazyArrays._datasize(view(x, 3:5)) == (3,)
            @test LazyArrays._datasize(transpose(x)) == (1, 3)
            
            arr = (1:10, cache(1:10))
            @test LazyArrays.max_datasize(LazyArrays._datasizes(arr)) == (10,)
            
            arr = (cache(1:10), cache(1:10))
            resizedata!(arr[1], 3)
            @test LazyArrays.max_datasize(LazyArrays._datasizes(arr)) == (3,)
            resizedata!(arr[2], 7)
            @test LazyArrays.max_datasize(LazyArrays._datasizes(arr)) == (7,)
            
            arr = (rand(10, 5), LazyArrays.CachedArray(rand(10, 5)))
            @test LazyArrays.max_datasize(LazyArrays._datasizes(arr)) == (10, 5)
            
            arr = (LazyArrays.CachedArray(rand(10, 5)), LazyArrays.CachedArray(rand(10, 5)))
            @test LazyArrays.max_datasize(LazyArrays._datasizes(arr)) == (0, 0)
            resizedata!(arr[1], 3, 4)
            @test LazyArrays.max_datasize(LazyArrays._datasizes(arr)) == (3, 4)
            resizedata!(arr[2], 2, 5);
            @test LazyArrays.max_datasize(LazyArrays._datasizes(arr)) == (3, 5) 

            arr = (1, [1, 2])
            @test LazyArrays._datasize(1) == (1, )
            @test LazyArrays.max_datasize(LazyArrays._datasizes(arr)) == (2,)
        end

        @testset "conforming_resize!" begin
            args = (cache(1:10), cache(1:10));
            LazyArrays.conforming_resize!(args);
            @test LazyArrays._datasizes(args) == ((0,), (0,));
            LazyArrays.resizedata!(args[2], 4);
            LazyArrays.conforming_resize!(args);
            @test LazyArrays._datasizes(args) == ((4,), (4,));
            
            args = (1:10, cache(1:10));
            LazyArrays.conforming_resize!(args);
            @test LazyArrays._datasizes(args) == ((10,), (10,));
            
            args = (cache(1:10)', cache(1:10)');
            LazyArrays.conforming_resize!(args);
            @test LazyArrays._datasizes(args) == ((0, 0), (0, 0));
            LazyArrays.resizedata!(args[1], 1, 3);
            LazyArrays.conforming_resize!(args);
            @test LazyArrays._datasizes(args) == ((1, 3), (1, 3));
            
            args = (cache(1:10)', LazyArrays.CachedArray(rand(1, 10)));
            @test LazyArrays._datasizes(args) == ((0, 0), (0, 0));
            LazyArrays.conforming_resize!(args);
            @test LazyArrays._datasizes(args) == ((0, 0), (0, 0));
            LazyArrays.resizedata!(args[1], 1, 4);
            LazyArrays.conforming_resize!(args);
            @test LazyArrays._datasizes(args) == ((1, 4), (1, 4));
            LazyArrays.resizedata!(args[2], 1, 6)
            LazyArrays.conforming_resize!(args);
            @test LazyArrays._datasizes(args) == ((1, 6), (1, 6));
            
            args = (cache(1:10), LazyArrays.CachedArray(rand(1, 10))');
            LazyArrays.resizedata!(args[1], 4);
            LazyArrays.conforming_resize!(args);
            @test LazyArrays._datasizes(args) == ((4,), (4, 1));
            
            args = (cache(1:10), LazyArrays.CachedArray(rand(1, 10))', 1:10);
            LazyArrays.conforming_resize!(args);
            @test LazyArrays._datasizes(args) == ((10,), (10, 1), (10,));

            args = (1, cache(1:2));
            LazyArrays.conforming_resize!(args);
            @test LazyArrays._datasizes(args) == ((1,), (1,)) # because scalars are treated as size (1,)

            args = (rand(10, 10), LazyArrays.CachedArray(rand(10, 10)));
            LazyArrays.conforming_resize!(args);
            @test LazyArrays._datasizes(args) == ((10, 10), (10, 10));

            args = (LazyArrays.CachedArray(rand(10, 10)), LazyArrays.CachedArray(rand(10, 10)));
            LazyArrays.conforming_resize!(args);
            @test LazyArrays._datasizes(args) == ((0, 0), (0, 0));
            LazyArrays.resizedata!(args[1], 3, 4);
            LazyArrays.conforming_resize!(args);
            @test LazyArrays._datasizes(args) == ((3, 4), (3, 4));
            LazyArrays.resizedata!(args[2], 5, 6);
            LazyArrays.conforming_resize!(args);
            @test LazyArrays._datasizes(args) == ((5, 6), (5, 6));

            @testset "conforming_resize! dimension mismatch" begin
                args = (cache(1:10), cache(1:10));
                @test LazyArrays.conforming_resize!(args) == ([], [])
                
                args = (cache(1:10), LazyArrays.CachedArray(rand(10, 5)));
                @test_throws ArgumentError LazyArrays.conforming_resize!(args)
                
                args = (LazyArrays.CachedArray(rand(3, 4)), LazyArrays.CachedArray(rand(4, 2)));
                @test_throws ArgumentError LazyArrays.conforming_resize!(args)

                args = (LazyArrays.CachedArray(rand(3, 4)), LazyArrays.CachedArray(rand(3, 4)));
                @test LazyArrays.conforming_resize!(args) == ([;;], [;;])
                LazyArrays.resizedata!(args[1], 2, 2);  
                @test LazyArrays.conforming_resize!(args) == (view(args[1].data, 1:2, 1:2), view(args[2].data, 1:2, 1:2))
                
                args = (1, cache(1:10));
                @test LazyArrays.conforming_resize!(args) == (1, [1])
                
                args = (cache(1:10), LazyArrays.CachedArray(rand(2, 3)), reshape(cache(1:8), 2, 2, 2));
                @test_throws ArgumentError LazyArrays.conforming_resize!(args)
                
                args = (reshape(cache(1:8), 2, 2, 2), reshape(cache(1:27), 3, 3, 3));
                @test_throws ArgumentError LazyArrays.conforming_resize!(args)
                
                args = ();
                @test LazyArrays.conforming_resize!(args) === args
                
                args = (cache(1:10),);
                @test LazyArrays.conforming_resize!(args) == ([],)
            end
        end
    end

    @testset "cacheddata for ApplyArray and BroadcastArray" begin
        x = ApplyArray(+, 1:10, cache(11:20));
        @test cacheddata(x) == ApplyArray(+, 1:10, 11:20) 
        @test Base.Broadcast.BroadcastStyle(typeof(cacheddata(x))) == LazyArrays.LazyArrayStyle{1}()
        x = view(x, 1:5);
        @test cacheddata(x) == ApplyArray(+, 1:5, 11:15)
        @test Base.Broadcast.BroadcastStyle(typeof(cacheddata(x))) == LazyArrays.LazyArrayStyle{1}()
        x = x';
        @test cacheddata(x) == transpose(ApplyArray(+, 1:5, 11:15))
        @test_broken Base.Broadcast.BroadcastStyle(typeof(cacheddata(x))) == LazyArrays.LazyArrayStyle{2}()
        @test Base.Broadcast.BroadcastStyle(typeof(cacheddata(x))) == Base.Broadcast.DefaultArrayStyle{2}()

        x = BroadcastVector(*, 1:10, cache(1:10));
        @test cacheddata(x) == BroadcastVector(*, 1:10, 1:10)
        @test Base.Broadcast.BroadcastStyle(typeof(cacheddata(x))) == LazyArrays.LazyArrayStyle{1}()
        x = view(x, 2:7);
        @test cacheddata(x) == BroadcastVector(*, 2:7, 2:7)
        @test_broken Base.Broadcast.BroadcastStyle(typeof(cacheddata(x))) == LazyArrays.LazyArrayStyle{1}()
        @test Base.Broadcast.BroadcastStyle(typeof(cacheddata(x))) == Base.Broadcast.DefaultArrayStyle{1}()
        x = x';
        @test cacheddata(x) == transpose(BroadcastVector(*, 2:7, 2:7))
        @test_broken Base.Broadcast.BroadcastStyle(typeof(cacheddata(x))) == LazyArrays.LazyArrayStyle{2}()
        @test Base.Broadcast.BroadcastStyle(typeof(cacheddata(x))) == Base.Broadcast.DefaultArrayStyle{2}()

        x = ApplyArray(+, cache(1:10), cache(11:20));
        @test cacheddata(x) == ApplyArray(+, 1:0, 1:0)
        @test Base.Broadcast.BroadcastStyle(typeof(cacheddata(x))) == LazyArrays.LazyArrayStyle{1}()
        LazyArrays.resizedata!(x.args[1], 3)
        @test cacheddata(x) == ApplyArray(+, 1:3, 11:13)
        @test Base.Broadcast.BroadcastStyle(typeof(cacheddata(x))) == LazyArrays.LazyArrayStyle{1}()
        x = x';
        @test cacheddata(x) == transpose(ApplyArray(+, 1:3, 11:13))
        @test Base.Broadcast.BroadcastStyle(typeof(cacheddata(x))) == LazyArrays.LazyArrayStyle{2}()
        x = view(x, 1:2);
        @test cacheddata(x) == [12, 14]
        @test Base.Broadcast.BroadcastStyle(typeof(cacheddata(x))) == LazyArrays.LazyArrayStyle{1}()

        x = BroadcastVector(*, cache(1:10), cache(11:20));
        @test cacheddata(x) == BroadcastVector(*, 1:0, 1:0)
        @test Base.Broadcast.BroadcastStyle(typeof(cacheddata(x))) == LazyArrays.LazyArrayStyle{1}()
        LazyArrays.resizedata!(x.args[1], 4)
        @test cacheddata(x) == BroadcastVector(*, 1:4, 11:14)
        @test Base.Broadcast.BroadcastStyle(typeof(cacheddata(x))) == LazyArrays.LazyArrayStyle{1}()
        x = x';
        @test cacheddata(x) == transpose(BroadcastVector(*, 1:4, 11:14))
        @test Base.Broadcast.BroadcastStyle(typeof(cacheddata(x))) == LazyArrays.LazyArrayStyle{2}()
    end
    =#            
    
    @testset "resizedata! for AdjTrans array" begin
        x = LazyArrays.CachedArray(rand(1, 10))';
        @test parent(x).datasize == (0, 0)
        LazyArrays.resizedata!(x, 4, 1)
        @test parent(x).datasize == (1, 4)
    end
end 

end # module
