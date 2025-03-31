module PaddedTests

using LazyArrays, FillArrays, ArrayLayouts, Base64, Test
using StaticArrays
import LazyArrays: PaddedLayout, PaddedRows, PaddedColumns, LayoutVector, MemoryLayout, paddeddata, ApplyLayout, sub_materialize, CachedVector, simplifiable
import ArrayLayouts: OnesLayout
import Base: setindex
using LinearAlgebra

# padded block arrays have padded data that is also padded. This is to test this
struct PaddedPadded <: LayoutVector{Int} end

MemoryLayout(::Type{PaddedPadded}) = PaddedColumns{UnknownLayout}()
Base.size(::PaddedPadded) = (10,)
Base.getindex(::PaddedPadded, k::Int) = k ≤ 5 ? 1 : 0
paddeddata(a::PaddedPadded) = a

@testset "Padded" begin
    @testset "Vcat" begin
        @test @inferred(MemoryLayout(typeof(Vcat(Ones(10),Zeros(10))))) == PaddedColumns{OnesLayout}()
        @test @inferred(MemoryLayout(typeof(Vcat([1.],Zeros(10))))) == PaddedColumns{DenseColumnMajor}()

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
            @test MemoryLayout(A) isa PaddedColumns{ScalarLayout}
            @test paddeddata(A) == 1
            B = Vcat(1, 2, Zeros(8))
            @test paddeddata(B) == [1,2]
            @test MemoryLayout(B) isa PaddedColumns{ApplyLayout{typeof(vcat)}}
            C = Vcat(1, cache(Zeros(8)));
            @test paddeddata(C) == [1]
            @test MemoryLayout(C) isa PaddedColumns{ApplyLayout{typeof(vcat)}}
            D = Vcat(1, 2, cache(Zeros(8)));
            @test paddeddata(D) == [1,2]
            @test MemoryLayout(D) isa PaddedColumns{ApplyLayout{typeof(vcat)}}
        end

        @testset "PaddedPadded" begin
            @test colsupport(PaddedPadded()) ≡ Base.OneTo(10)
            @test stringmime("text/plain", PaddedPadded()) == "10-element $PaddedPadded:\n 1\n 1\n 1\n 1\n 1\n 0\n 0\n 0\n 0\n 0"
            @test dot(PaddedPadded(), PaddedPadded()) == 5
            @test dot(PaddedPadded(), 1:10) == dot(1:10, PaddedPadded()) == 15

            @test PaddedPadded()[1:7] isa Vcat

            @test norm(PaddedPadded()) ≈ sqrt(5)
        end

        @testset "Matrix padded" begin
            a = Vcat(randn(2,3), Zeros(3,3))
            b = Vcat(randn(3,3), Zeros(2,3))
            @test MemoryLayout(a) isa PaddedColumns{DenseColumnMajor}
            @test a + b isa Vcat
            @test MemoryLayout(a+b) isa PaddedColumns{DenseColumnMajor}
            @test a + b == Matrix(a) + Matrix(b)
        end
    end

    @testset "copyto!" begin
        a = Vcat(1, Zeros(10));
        c = cache(Zeros(11));
        @test MemoryLayout(typeof(a)) isa PaddedColumns
        @test MemoryLayout(typeof(c)) isa PaddedColumns{DenseColumnMajor}
        @test copyto!(c, a) ≡ c;
        @test c.datasize[1] == 1
        @test c == a

        a = Vcat(1:3, Zeros(10))
        c = cache(Zeros(13));
        @test MemoryLayout(typeof(a)) isa PaddedColumns
        @test MemoryLayout(typeof(c)) isa PaddedColumns{DenseColumnMajor}
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
        @test a ≠ b
        b = cache(Zeros(13));
        a[3] = 2; b[3] = 2; b[5]=0;
        @test a == b

        a = Vcat([1 2; 3 4], Zeros(10,2))
        c = cache(Zeros(12, 2));
        @test copyto!(c, a) == c == a
        @test copyto!(c, Zeros(12,2)) == c == Zeros(12,2)

        # following not yet supported
        @test_throws ErrorException copyto!(c, Zeros(10,2))
    end

    @testset "vcat and padded" begin
        x,y = Vcat([1,2,3],Zeros(5)), Vcat(5, 1:7)
        @test MemoryLayout(x) isa PaddedColumns
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
        @test MemoryLayout(V) isa PaddedRows
        @test A[parentindices(V)...] == copy(V) == Array(A)[parentindices(V)...]
        V = view(A,3:5,1:4)
        @test MemoryLayout(V) isa PaddedRows
        @test @inferred(paddeddata(V)) == reshape(3:5,3,1)

        v = view(A,2,1:5)
        @test MemoryLayout(v) isa PaddedColumns
        @test paddeddata(v) == [2]
        @test A[2,1:5] == copy(v) == sub_materialize(v)

        @testset "Padded subarrays" begin
            a = Vcat([1,2,3],[4,5,6])
            @test sub_materialize(view(a,2:6)) == a[2:6]
            a = Vcat([1,2,3], Zeros(10))
            c = cache(Zeros(10)); c[1:3] = 1:3;
            v = view(a,2:4)
            w = view(c,2:4);
            @test MemoryLayout(typeof(a)) isa PaddedColumns{DenseColumnMajor}
            @test MemoryLayout(v) isa PaddedColumns{DenseColumnMajor}
            @test sub_materialize(v) == a[2:4] == sub_materialize(w)
            @test sub_materialize(v) isa Vcat
            @test sub_materialize(w) isa Vcat
            A = Vcat(Eye(2), Zeros(10,2))
            @test MemoryLayout(A) isa PaddedColumns
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
        @test MemoryLayout(H) isa PaddedRows
        @test paddeddata(H) == [1 3]
        @test rowsupport(H) == 1:2

        @test MemoryLayout(Hcat(1, H)) isa PaddedRows
        @test paddeddata(Hcat(1, H)) == [1 1 3]
        @test MemoryLayout(Hcat(1, 2, H)) isa PaddedRows
        @test paddeddata(Hcat(1, 2, H)) == [1 2 1 3]
    end
    @testset "padded broadcast" begin
        @testset "vector" begin
            a = Vcat([1], Zeros(3))
            b = Vcat([1,2],Zeros(2))
            v = Vcat(1, 1:3)
            n = Vcat(1, Zeros{Int}(3))
            c = cache(Zeros(4));
            @test @inferred(a + b) isa Vcat
            @test @inferred(a .* b) isa Vcat
            @test paddeddata(a .* b) == [1]
            @test a + v == v + a
            @test a .* v == v .* a

            @test n .+ n ≡ Vcat(2,Zeros{Int}(3))
            @test n .+ v ≡ n .+ v
            @test n .+ v ≡ v .+ n ≡ Vcat(2,1:3)
            @test n .+ b == b .+ n
            @test n .+ a == a .+ n
            @test n .+ c == c .+ n
            @test c.datasize == (0,)

            @test @inferred(n .* n) ≡ @inferred(n .* n) ≡ n
            @test @inferred(n .* a) ≡ @inferred(a .* n) ≡ Vcat(1,Zeros{Float64}(3))
            @test @inferred(n .* v) ≡ @inferred(v .* n) ≡ Vcat(1,Zeros{Int}(3))
            @test n .* c == c .* n

            @test view(a, 1:4) .* v == v .* view(a, 1:4)
        end
        @testset "matrix" begin
            a = Vcat([1 2], Zeros(3,2))
            b = Vcat([1 2; 3 4], Zeros(2,2))
            @test MemoryLayout(a + a) isa PaddedColumns
            @test a + a isa Vcat
            @test a + a == 2a
            @test a + b == b + a

            c = PaddedArray(randn(2,2), 4, 2)
            d = PaddedArray(randn(1,2), 4, 2)
            e = PaddedArray(randn(3,1), 4, 2)
            @test c + c == 2c
            @test c + a == a + c
            @test c + b == b + c
            @test c + d == d + c
            @test d + a == a + d
            @test d + b == b + d
            @test e + a == a + e
            @test e + b == b + e
            @test e + c == c + e
            @test e + d == d + e
        end
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

        @test MemoryLayout(P[1:6,1:7]) isa PaddedLayout
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
        @test MemoryLayout(a) isa PaddedColumns{DenseColumnMajor}
        @test paddeddata(a) == 1:2

        a = ApplyArray(setindex, Zeros(5,5), [1 2 3; 4 5 6], Base.OneTo(2), Base.OneTo(3))
        @test a == setindex!(zeros(5,5),[1 2 3; 4 5 6], 1:2, 1:3)
        @test MemoryLayout(a) isa PaddedLayout{DenseColumnMajor}
        @test paddeddata(a) == [1 2 3; 4 5 6]

        # need to add bounds checking
        @test_broken (try ApplyArray(setindex, Zeros(5,5), [1 2; 4 5], 2:3, 3:6) catch e; e; end) isa BoundsError

        @test PaddedArray(1, 3) == PaddedVector(1,3) == [1; zeros(2)]
        @test PaddedArray(1, 3, 3) == PaddedMatrix(1, 3, 3) == [1 zeros(1,2); zeros(2,3)]
        @test PaddedVector([1, 2, 3], 3:5)[3:5] == [3; 0; 0]
        @test PaddedMatrix([1 2; 3 4], (1:3, 1:3)) == [1 2 0; 3 4 0; 0 0 0]
    end

    @testset "adjtrans" begin
        a = Vcat(1, Zeros(3))
        @test MemoryLayout(a') isa DualLayout{<:PaddedRows}
        @test MemoryLayout(transpose(a)) isa DualLayout{<:PaddedRows}
        @test paddeddata(a') ≡ 1
        b = Vcat(SVector(1,2), Zeros(3))
        @test paddeddata(b') ≡ SVector(1,2)'
        @test paddeddata(transpose(b)) ≡ transpose(SVector(1,2))

        H = Hcat(1, 3, Zeros(1,3))
        @test MemoryLayout(Transpose(H)) isa PaddedColumns
        @test paddeddata(Transpose(H)) == [1,3]
    end

    @testset "norm" begin
        a = Vcat(1, Zeros(3))
        c = cache(Zeros(4)); c[1] = 1
        @test norm(a) ≡ LinearAlgebra.normInf(c) ≡ LinearAlgebra.norm2(c) ≡ LinearAlgebra.norm1(c) ≡ LinearAlgebra.normp(c,2) ≡ 1.0
    end

    @testset "padded columns" begin
        A = randn(5,5)
        U = UpperTriangular(A)
        v = view(U,:,3)
        @test MemoryLayout(v) isa PaddedColumns{DenseColumnMajor}
        @test layout_getindex(v,1:4) == U[1:4,3]
        @test layout_getindex(v,1:4) isa Vcat

        L = LowerTriangular(A)
        w = view(L,3,:)
        @test MemoryLayout(w) isa PaddedColumns{ArrayLayouts.StridedLayout}
        @test layout_getindex(w,1:4) == L[3,1:4]
        @test layout_getindex(w,1:4) isa Vcat
    end

    @testset "vcat sub arguments" begin
        a = Vcat(1:5, Zeros(10))
        @test LazyArrays.arguments(vcat, view(a, 1:7)) == (1:5, Zeros(2))
    end

    @testset "vcat padded" begin
        A = Vcat([1,2,3], Zeros(7))
        B = Vcat([1,2], Zeros(8))
        C = Vcat(A,B)
        D = Hcat(A', B')
        @test MemoryLayout(C) isa PaddedColumns
        @test paddeddata(C) == [A; 1:2]
        @test MemoryLayout(D) isa DualLayout{<:PaddedRows}
        @test paddeddata(D) == [A' (1:2)']

        E = Hcat(Hcat(randn(3,2), Zeros(3,3)), Hcat(randn(3,2), Zeros(3,3)))
        @test MemoryLayout(E) isa PaddedRows
    end

    @testset "scalar vcat bug" begin
        v = Vcat(1, Zeros(10))
        w = Vcat(1, 2, Zeros(9))
        @test v[1:10] == [1; zeros(9)]
        @test v[2:10] == zeros(9)
        @test w[1:10] == [1; 2; zeros(8)]
        @test w[2:10] == [2; zeros(8)]
        @test w[3:10] == zeros(8)
        H = Hcat(1, Zeros(1, 10))
        @test H[:,1:10] == [1 zeros(9)']
        @test H[:,2:10] == zeros(9)'
    end

    @testset "Mul simplifiable" begin
        a = Vcat(5, 1:7)
        b = Vcat([1,2], Zeros(6))
        @test a'b == b'a == Vector(a)'b
        @test simplifiable(*, a', b) == Val(true)
        @test simplifiable(*, b', a) == Val(true)

        D = Diagonal(Fill(2,8))
        @test D*b isa Vcat
        @test simplifiable(*, D, b) == Val(true)

        B = BroadcastArray(+, 1:8, (2:9)')
        C = ApplyArray(exp, randn(8,8))
        @test B'b == Matrix(B)'b
        @test b'B == b'Matrix(B)
        @test simplifiable(*, B', b) == Val(true)
        @test simplifiable(*, b', B) == Val(true)
        @test simplifiable(*, C', b) == Val(false)
        @test simplifiable(*, b', C) == Val(false)

        @test C'b ≈ Matrix(C)'b
        @test b'C ≈ b'Matrix(C)
    end

    @testset "Bidiagonal" begin
        B = Bidiagonal(1:5, 1:4, :L)
        b = Vcat(randn(5), Zeros(0))
        @test ArrayLayouts.ldiv!(B, deepcopy(b)) ≈ B\b
        c = cache(Zeros(5)); c[1] = 2;
        @test ArrayLayouts.ldiv!(B, c) ≈ B\[2; zeros(4)]

        c = cache(Zeros(5)); c[1:2] = [1,2];
        @test_throws SingularException ArrayLayouts.ldiv!(Bidiagonal(0:4, 1:4, :L), c)
        @test_throws SingularException ArrayLayouts.ldiv!(Bidiagonal(-1:3, 1:4, :L), c)
        @test_throws SingularException ArrayLayouts.ldiv!(Bidiagonal(-4:0, 1:4, :L), c)
    end

    @testset "Broadcast * Padded" begin
        B = BroadcastArray(*, 1:8, (2:9)')
        p = Vcat(1:2, Zeros(6))
        @test B*p == Matrix(B)*p
        @test simplifiable(*,B,p) == Val(true)
    end
end
end # module
