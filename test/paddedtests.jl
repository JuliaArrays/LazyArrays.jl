using LazyArrays, FillArrays, ArrayLayouts, StaticArrays, Base64, Test
import LazyArrays: PaddedLayout, LayoutVector, MemoryLayout, paddeddata, ApplyLayout, sub_materialize, CachedVector
import Base: setindex

# padded block arrays have padded data that is also padded. This is to test this
struct PaddedPadded <: LayoutVector{Int} end

MemoryLayout(::Type{PaddedPadded}) = PaddedLayout{UnknownLayout}()
Base.size(::PaddedPadded) = (10,)
Base.getindex(::PaddedPadded, k::Int) = k ≤ 5 ? 1 : 0
paddeddata(a::PaddedPadded) = a

@testset "Padded" begin
    @testset "PaddedLayout" begin
        A = Vcat([1,2,3], Zeros(7))
        B = Vcat([1,2], Zeros(8))

        C = @inferred(A+B)
        @test C isa Vcat{Float64,1}
        @test C.args[1] isa Vector
        @test C.args[2] isa Zeros{Float64}
        @test C == Vector(A) + Vector(B)

        @test colsupport(A, 1) == 1:3

        B = Vcat([1,2], Ones(8))

        C = @inferred(A+B)
        @test C isa Vcat{Float64,1}
        @test C.args[1] isa Vector{Float64}
        @test C.args[2] isa Ones{Float64}
        @test C == Vector(A) + Vector(B)

        B = Vcat([1,2], randn(8))

        C = @inferred(A+B)
        @test C isa Vcat
        @test C == Vector(A) + Vector(B)

        B = Vcat(SVector(1,2), Ones(8))
        C = @inferred(A+B)
        @test C isa Vcat{Float64,1}
        @test C.args[1] isa Vector{Float64}
        @test C.args[2] isa Ones{Float64}
        @test C == Vector(A) + Vector(B)


        A = Vcat(SVector(3,4), Zeros(8))
        B = Vcat(SVector(1,2), Ones(8))
        C = @inferred(A+B)
        @test C isa Vcat{Float64,1}
        @test C.args[1] isa SVector{2,Int}
        @test C.args[2] isa Ones{Float64}
        @test C == Vector(A) + Vector(B)

        @testset "multiple scalar" begin
            # We only do 1 or 2 for now, this should be redesigned later
            A = Vcat(1, Zeros(8))
            @test MemoryLayout(A) isa PaddedLayout{ScalarLayout}
            @test paddeddata(A) == 1
            B = Vcat(1, 2, Zeros(8))
            @test paddeddata(B) == [1,2]
            @test MemoryLayout(B) isa PaddedLayout{ApplyLayout{typeof(vcat)}}
            C = Vcat(1, cache(Zeros(8)));
            @test paddeddata(C) == [1]
            @test MemoryLayout(C) isa PaddedLayout{ApplyLayout{typeof(vcat)}}
            D = Vcat(1, 2, cache(Zeros(8)));
            @test paddeddata(D) == [1,2]
            @test MemoryLayout(D) isa PaddedLayout{ApplyLayout{typeof(vcat)}}
        end

        @testset "PaddedPadded" begin
            @test colsupport(PaddedPadded()) ≡ Base.OneTo(10)
            @test stringmime("text/plain", PaddedPadded()) == "10-element PaddedPadded:\n 1\n 1\n 1\n 1\n 1\n 0\n 0\n 0\n 0\n 0"
            @test dot(PaddedPadded(), PaddedPadded()) == 5
            @test dot(PaddedPadded(), 1:10) == dot(1:10, PaddedPadded()) == 15
        end
    end

    @testset "copyto!" begin
        a = Vcat(1, Zeros(10));
        c = cache(Zeros(11));
        @test MemoryLayout(typeof(a)) isa PaddedLayout
        @test MemoryLayout(typeof(c)) isa PaddedLayout{DenseColumnMajor}
        @test copyto!(c, a) ≡ c;
        @test c.datasize[1] == 1
        @test c == a

        a = Vcat(1:3, Zeros(10))
        c = cache(Zeros(13));
        @test MemoryLayout(typeof(a)) isa PaddedLayout
        @test MemoryLayout(typeof(c)) isa PaddedLayout{DenseColumnMajor}
        @test copyto!(c, a) ≡ c;
        @test c.datasize[1] == 3
        @test c == a

        @test dot(a,a) ≡ dot(a,c) ≡ dot(c,a) ≡ dot(c,c) ≡ 14.0


        a = Vcat(1:3, Zeros(5))
        c = cache(Zeros(13));
        @test copyto!(c, a) ≡ c;
        @test c.datasize[1] == 3
        @test c[1:8] == a

        a = cache(Zeros(13)); b = cache(Zeros(15));
        @test a ≠ b
        b = cache(Zeros(13));
        a[3] = 2; b[3] = 2; b[5]=0;
        @test a == b
    end

    @testset "vcat and padded" begin
        x,y = Vcat([1,2,3],Zeros(5)), Vcat(5, 1:7)
        @test MemoryLayout(x) isa PaddedLayout
        @test MemoryLayout(y) isa ApplyLayout{typeof(vcat)}
        @test x .+ y == y .+ x == Vector(x) .+ Vector(y)
        @test x .+ y isa Vcat
        @test y .+ x isa Vcat

        c = cache(Zeros(8));
        @test c + x == x + c
        @test c + y == y + c
    end

    @testset "vcat and Zeros" begin
        x,y = Vcat([1,2,3],Zeros(5)), Vcat(5, 1:7)
        @test x .+ Zeros(8) == Zeros(8) .+ x == x
        @test y .+ Zeros(8) == Zeros(8) .+ y == y
        @test x .* Zeros(8) ≡ Zeros(8) .* x ≡ Zeros(8)
        @test y .* Zeros(8) ≡ Zeros(8) .* y ≡ Zeros(8)
        @test x .\ Zeros(8) ≡ Zeros(8) ./ x ≡ Zeros(8)
        @test y .\ Zeros(8) ≡ Zeros(8) ./ y ≡ Zeros(8)
    end

    @testset "vcat and BroadcastArray" begin
        x,y,z = Vcat([1,2,3],Zeros(5)), Vcat(5, 1:7), BroadcastArray(exp,1:8)
        @test x .+ z == z .+ x == Array(x) .+ Array(z)
        @test y .+ z == z .+ y == Array(y) .+ Array(z)

        @test x .+ z isa CachedVector
        @test y .+ z isa BroadcastVector
    end
    @testset "subpadded" begin
        A = Hcat(1:10, Zeros(10,10))
        V = view(A,3:5,:)
        @test MemoryLayout(V) isa PaddedLayout
        @test A[parentindices(V)...] == copy(V) == Array(A)[parentindices(V)...]
        V = view(A,3:5,1:4)
        @test MemoryLayout(V) isa PaddedLayout
        @test @inferred(paddeddata(V)) == reshape(3:5,3,1)

        v = view(A,2,1:5)
        @test MemoryLayout(v) isa PaddedLayout
        @test paddeddata(v) == [2]
        @test A[2,1:5] == copy(v) == sub_materialize(v)

        @testset "Padded subarrays" begin
            a = Vcat([1,2,3],[4,5,6])
            @test sub_materialize(view(a,2:6)) == a[2:6]
            a = Vcat([1,2,3], Zeros(10))
            c = cache(Zeros(10)); c[1:3] = 1:3;
            v = view(a,2:4)
            w = view(c,2:4);
            @test MemoryLayout(typeof(a)) isa PaddedLayout{DenseColumnMajor}
            @test MemoryLayout(v) isa PaddedLayout{DenseColumnMajor}
            @test sub_materialize(v) == a[2:4] == sub_materialize(w)
            @test sub_materialize(v) isa Vcat
            @test sub_materialize(w) isa Vcat
            A = Vcat(Eye(2), Zeros(10,2))
            V = view(A, 1:5, 1:2)
            @test sub_materialize(V) == A[1:5,1:2]
        end
    end
    @testset "hcat" begin
        z = cache(Zeros(1,3));
        z[1] = 3;
        H = Hcat([1 2], z);
        @test MemoryLayout(H) isa PaddedLayout
        @test paddeddata(H) == [1 2 3]
        H = Hcat(1, 3, Zeros(1,3))
        @test MemoryLayout(H) isa PaddedLayout
        @test paddeddata(H) == [1 3]
    end
    @testset "padded broadcast" begin
        a = Vcat([1], Zeros(3))
        b = Vcat([1,2],Zeros(2))
        v = Vcat(1, 1:3)
        @test a + b isa Vcat
        @test a .* b isa Vcat
        @test paddeddata(a .* b) == [1]
        @test a + v == v + a
        @test a .* v == v .* a
    end

    @testset "hvcat" begin
        P = ApplyArray(hvcat, 2, randn(4,5), Zeros(4,6), Zeros(6,5), Zeros(6,6))
        @test eltype(P) == Float64
        @test MemoryLayout(P) isa PaddedLayout
        @test P == hvcat(P.args...)
        @test @inferred(colsupport(P,3)) == Base.OneTo(4)
        @test @inferred(colsupport(P,7)) == Base.OneTo(0)
        @test @inferred(rowsupport(P,3)) == Base.OneTo(5)
        @test @inferred(rowsupport(P,7)) == Base.OneTo(0)
        @test copyto!(similar(P), P) == P

        @test P[3,:] isa Vcat
        @test P[3,1:11] == P[3,:]
        @test P[:,3] isa Vcat
        @test P[1:10,3] == P[:,3]
        @test P[6,:] isa Vcat
        @test P[1:10,6] == P[:,6]
        @test P[:,6] isa Vcat
        @test P[6,1:11] == P[6,:]
    end
    @testset "setindex" begin
        a = ApplyArray(setindex, 1:6, 5, 2)
        @test a == [1; 5; 3:6]
        @test_throws BoundsError a[7]
        a = ApplyArray(setindex, 1:6, [9,8,7], 1:3)
        @test a == [9; 8; 7; 4:6]
        @test_throws BoundsError a[7]

        a = ApplyArray(setindex, Zeros(5,5), 2, 2, 3)
        @test a[2,3] === 2.0
        @test a == setindex!(zeros(5,5),2,2,3)

        a = ApplyArray(setindex, Zeros(5,5), [4,5], 2:3, 3)
        @test a == setindex!(zeros(5,5),[4,5], 2:3, 3)

        a = ApplyArray(setindex, Zeros(5,5), [1 2 3; 4 5 6], 2:3, 3:5)
        @test a == setindex!(zeros(5,5),[1 2 3; 4 5 6], 2:3, 3:5)


        a = ApplyArray(setindex, Zeros(5), [1,2], Base.OneTo(2))
        @test MemoryLayout(a) isa PaddedLayout{DenseColumnMajor}
        @test paddeddata(a) == 1:2

        a = ApplyArray(setindex, Zeros(5,5), [1 2 3; 4 5 6], Base.OneTo(2), Base.OneTo(3))
        @test a == setindex!(zeros(5,5),[1 2 3; 4 5 6], 1:2, 1:3)
        @test MemoryLayout(a) isa PaddedLayout{DenseColumnMajor}
        @test paddeddata(a) == [1 2 3; 4 5 6]

        # need to add bounds checking
        @test_broken ApplyArray(setindex, Zeros(5,5), [1 2; 4 5], 2:3, 3:5)

        @test PaddedArray(1, 3) == PaddedVector(1,3) == [1; zeros(2)]
        @test PaddedArray(1, 3, 3) == PaddedMatrix(1, 3, 3) == [1 zeros(1,2); zeros(2,3)]
    end

    @testset "adjtrans" begin
        a = Vcat(1, Zeros(3))
        @test MemoryLayout(a') isa DualLayout{<:PaddedLayout}
        @test MemoryLayout(transpose(a)) isa DualLayout{<:PaddedLayout}
        @test paddeddata(a') ≡ 1
        b = Vcat(SVector(1,2), Zeros(3))
        @test paddeddata(b') ≡ SVector(1,2)'
        @test paddeddata(transpose(b)) ≡ transpose(SVector(1,2))
    end

    @testset "norm" begin
        a = Vcat(1, Zeros(3))
        c = cache(Zeros(4)); c[1] = 1
        @test norm(a) ≡ LinearAlgebra.normInf(c) ≡ LinearAlgebra.norm2(c) ≡ LinearAlgebra.norm1(c) ≡ LinearAlgebra.normp(c,2) ≡ 1.0
    end
end