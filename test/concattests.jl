module ConcatTests

using LazyArrays, FillArrays, LinearAlgebra, ArrayLayouts, Test, Base64
using StaticArrays
import LazyArrays: MemoryLayout, DenseColumnMajor, materialize!, call, paddeddata,
                    MulAdd, Applied, ApplyLayout, DefaultApplyStyle, sub_materialize, resizedata!,
                    CachedVector, ApplyLayout, arguments, BroadcastVector, LazyLayout

@testset "concat" begin
    @testset "Vcat" begin
        @testset "Vector" begin
            A = @inferred(Vcat(Vector(1:10), Vector(1:20)))
            @test eltype(A) == Int
            @test A == ApplyArray(vcat, Vector(1:10), Vector(1:20))
            @test @inferred(axes(A)) == (Base.OneTo(30),)
            @test @inferred(A[5]) == A[15] == 5
            @test_throws BoundsError A[31]
            @test reverse(A) == Vcat(Vector(reverse(1:20)), Vector(reverse(1:10)))
            b = Array{Int}(undef, 31)
            @test_throws DimensionMismatch copyto!(b, A)
            b = Array{Int}(undef, 30)
            @test @allocated(copyto!(b, A)) == 0
            @test b == vcat(A.args...)
            @test copy(A) isa Vcat
            @test copy(A) == A
            @test copy(A) !== A
            @test vec(A) === A
            @test A' == transpose(A) == Vector(A)'
            @test permutedims(A) == permutedims(Vector(A))

            A = @inferred(Vcat(1:10, 1:20))
            @test @inferred(length(A)) == 30
            @test @inferred(A[5]) == A[15] == 5
            @test_throws BoundsError A[31]
            @test reverse(A) == Vcat(reverse(1:20), reverse(1:10))
            b = Array{Int}(undef, 31)
            @test_throws DimensionMismatch copyto!(b, A)
            b = Array{Int}(undef, 30)
            copyto!(b, A)
            @test_broken @allocated(copyto!(b, A)) == 0
            @test @allocated(copyto!(b, A)) ≤ 200
            @test b == vcat(A.args...)
            @test copy(A) === A
            @test vec(A) === A
            @test A' == transpose(A) == Vector(A)'
            @test A' == Hcat((1:10)', (1:20)')
            @test transpose(A) == Hcat(transpose(1:10), transpose(1:20))
            @test permutedims(A) == permutedims(Vector(A))

            @test map(copy,A) isa Vcat
            @test Applied(A)[3] == 3
        end

        @testset "Matrix" begin
            A = Vcat(randn(2,10), randn(4,10))
            @test @inferred(length(A)) == 60
            @test @inferred(size(A)) == (6,10)
            @test_throws BoundsError A[61]
            @test_throws BoundsError A[7,1]
            b = Array{Float64}(undef, 7,10)
            @test_throws DimensionMismatch copyto!(b, A)
            b = Array{Float64}(undef, 6,10)
            @test_broken @allocated(copyto!(b, A)) == 0
            @test @allocated(copyto!(b, A)) ≤ 200
            @test b == vcat(A.args...)
            @test copy(A) isa Vcat
            @test copy(A) == A
            @test copy(A) !== A
            @test vec(A) == vec(Matrix(A))
            @test A' == transpose(A) == Matrix(A)'
            @test permutedims(A) == permutedims(Matrix(A))
            @test_throws BoundsError A[7,2] = 6
            @test Applied(A)[1,3] == A[1,3]

            A = Vcat(randn(2,10).+im.*randn(2,10), randn(4,10).+im.*randn(4,10))
            @test eltype(A) == ComplexF64
            @test @inferred(length(A)) == 60
            @test @inferred(size(A)) == (6,10)
            @test_throws BoundsError A[61]
            @test_throws BoundsError A[7,1]
            b = Array{ComplexF64}(undef, 7,10)
            @test_throws DimensionMismatch copyto!(b, A)
            b = Array{ComplexF64}(undef, 6,10)
            @test_broken @allocated(copyto!(b, A)) == 0
            @test @allocated(copyto!(b, A)) ≤ 200
            @test b == vcat(A.args...)
            @test copy(A) isa Vcat
            @test copy(A) == A
            @test copy(A) !== A
            @test vec(A) == vec(Matrix(A))
            @test A' == Matrix(A)'
            @test transpose(A) == transpose(Matrix(A))
            @test permutedims(A) == permutedims(Matrix(A))

            @testset "indexing" begin
                A = Vcat(randn(2,10), randn(4,10))
                @test A[2,1:5] == Matrix(A)[2,1:5]
                @test A[2,:] == Matrix(A)[2,:]
                @test A[1:5,2] == Matrix(A)[1:5,2]
                @test A[:,2] == Matrix(A)[:,2]
                @test A[:,:] == A[1:6,:] == A[:,1:10] == A[1:6,1:10] == A
            end
        end

        @testset "etc" begin
            @test Vcat() isa Vcat{Any,1,Tuple{}}

            A = Vcat(1,zeros(3,1))
            @test A isa AbstractMatrix
            @test A[1,1] == 1.0
            @test A[2,1] == 0.0
            @test axes(A) == (Base.OneTo(4),Base.OneTo(1))
            @test permutedims(A) == permutedims(Matrix(A))
        end

        @testset "Vcat adjoints of vectors" begin
            # This special case was added to catch fast paths but
            # could be removed
            v = Vcat((1:5)', (2:6)')
            @test copyto!(Matrix{Float64}(undef,2,5), v) == Matrix(v) == [(1:5)'; (2:6)']
        end

        @testset "adjoint sub" begin
            @test arguments(view(Vcat(1,1:10)',1,:)) == (Fill(1,1),1:10)
        end

        @testset "Empty Vcat" begin
            @test @inferred(Vcat{Int}([1])) == [1]
            @test @inferred(Vcat{Int}()) == Int[]
        end

        @testset "in" begin
            @test 1 in Vcat(1, 1:10_000_000_000)
            @test 100_000_000 in Vcat(1, 1:10_000_000_000)
        end

        @testset "convert" begin
            for T in (Float32, Float64, ComplexF32, ComplexF64)
                Z = Vcat(zero(T),Zeros{T}(10))
                @test convert(AbstractArray,Z) ≡ Z
                @test convert(AbstractArray{T},Z) ≡ AbstractArray{T}(Z) ≡ Z
                @test convert(AbstractVector{T},Z) ≡ AbstractVector{T}(Z) ≡ Z
            end
        end

         @testset "Any/All" begin
            @test all(Vcat(true, Fill(true,100_000_000)))
            @test any(Vcat(false, Fill(true,100_000_000)))
            @test all(iseven, Vcat(2, Fill(4,100_000_000)))
            @test any(iseven, Vcat(2, Fill(1,100_000_000)))
            @test_throws TypeError all(Vcat(1))
            @test_throws TypeError any(Vcat(1))
        end

        @testset "isbitsunion #45" begin
            @test copyto!(Vector{Vector{Int}}(undef,6), Vcat([[1], [2], [3]], [[1], [2], [3]])) ==
                [[1], [2], [3], [1], [2], [3]]

            a = Vcat{Union{Float64,UInt8}}([1.0], [UInt8(1)])
            @test Base.isbitsunion(eltype(a))
            r = Vector{Union{Float64,UInt8}}(undef,2)
            @test copyto!(r, a) == a
            @test r == a
            @test copyto!(Vector{Float64}(undef,2), a) == [1.0,1.0]
        end

         @testset "maximum/minimum Vcat" begin
            x = Vcat(1:2, [1,1,1,1,1], 3)
            @test maximum(x) == 3
            @test minimum(x) == 1
        end


        @testset "searchsorted" begin
            a = Vcat(1:1_000_000, [10_000_000_000,12_000_000_000], 14_000_000_000)
            b = Vcat(1, 3:1_000_000, [2_000_000, 3_000_000])
            @test @inferred(searchsortedfirst(a, 6_000_000_001)) == 1_000_001
            @test @inferred(searchsortedlast(a, 2)) == 2
            @test @inferred(searchsortedfirst(b, 5)) == 4
            @test @inferred(searchsortedlast(b, 1)) == 1
        end

        @testset "args with hcat and view" begin
            A = Vcat(fill(2.0,1,10),ApplyArray(hcat, Zeros(1), fill(3.0,1,9)))
            @test arguments(view(A,:,10)) == ([2.0], [3.0])
        end

        @testset "union" begin
            a = Vcat([1,3,4],5:7)
            b = Vcat([1,3,4],5:7)
            union(a,b)
        end

        @testset "==" begin
            A = Vcat([1,2],[0])
            B = Vcat([1,2],[0])
            C = Vcat([1],[2,0])
            @test A == B == C == [1,2,0]
            @test A ≠ [1,2,4]
        end

        @testset "resizedata!" begin
            # allow emulating a cached Vector
            a = Vcat([1,2], Zeros(8))
            @test resizedata!(a, 2) ≡ a
            @test_throws BoundsError resizedata!(a,3)
        end

        @testset "Axpy" begin
            a = Vcat([1.,2],Zeros(1_000_000))
            b = Vcat([1.,2],Zeros(1_000_000))
            axpy!(2.0, a, b)
            @test b[1:10] == [3; 6; zeros(8)]
            axpy!(2.0, view(a,:), b)
            @test b[1:10] == [5; 10; zeros(8)]
        end

        @testset "l/rmul!" begin
            a = Vcat([1.,2],Zeros(1_000_000))
            @test ArrayLayouts.lmul!(2,a) ≡ a
            @test a[1:10] == [2; 4; zeros(8)]
            @test ArrayLayouts.rmul!(a,2) ≡ a
            @test a[1:10] == [4; 8; zeros(8)]
        end

        @testset "Dot" begin
            a = Vcat([1,2],Zeros(1_000_000))
            b = Vcat([1,2,3],Zeros(1_000_000))
            @test @inferred(dot(a,b)) ≡ 5.0
            @test @inferred(dot(a,1:1_000_002)) ≡ @inferred(dot(1:1_000_002,a)) ≡ 5.0
        end

        @testset "search" begin
            a = Vcat([1,2], 5:100)
            v = Vector(a)
            @test searchsortedfirst(a, 0) ≡ searchsortedfirst(v, 0) ≡ 1
            @test searchsortedfirst(a, 2) ≡ searchsortedfirst(v, 2) ≡ 2
            @test searchsortedfirst(a, 4) ≡ searchsortedfirst(v, 4) ≡ 3
            @test searchsortedfirst(a, 50) ≡ searchsortedfirst(v, 50) ≡ 48
            @test searchsortedfirst(a, 101) ≡ searchsortedfirst(v, 101) ≡ 99
            @test searchsortedlast(a, 0) ≡ searchsortedlast(v, 0) ≡ 0
            @test searchsortedlast(a, 2) ≡ searchsortedlast(v, 2) ≡ 2
            @test searchsortedlast(a, 4) ≡ searchsortedlast(v, 4) ≡ 2
            @test searchsortedlast(a, 50) ≡ searchsortedlast(v, 50) ≡ 48
            @test searchsortedlast(a, 101) ≡ searchsortedlast(v, 101) ≡ 98
            @test searchsorted(a, 0) ≡ searchsorted(v, 0) ≡ 1:0
            @test searchsorted(a, 2) ≡ searchsorted(v, 2) ≡ 2:2
            @test searchsorted(a, 4) ≡ searchsorted(v, 4) ≡ 3:2
            @test searchsorted(a, 50) ≡ searchsorted(v, 50) ≡ 48:48
            @test searchsorted(a, 101) ≡ searchsorted(v, 101) ≡ 99:98
        end

        @testset "print" begin
            @test Base.replace_in_print_matrix(Vcat(1:3,Zeros(10)), 4, 1, "0.0") == " ⋅ "
        end

        @testset "projection" begin
            a = Vcat(Ones(5), Zeros(5))
            b = randn(10)
            @test colsupport(a .* b) ≡ Base.OneTo(5)
            @test a .* b isa Vcat
            @test Diagonal(a) * b isa Vcat
            @test b .* a isa Vcat
            @test a .* b == Diagonal(a) * b == b .* a
        end
    end

    @testset "Hcat" begin
        A = @inferred(Hcat(1:10, 2:11))
        @test_throws BoundsError A[1,3]
        @test_throws BoundsError A[11,1]

        @test @inferred(call(A)) == hcat
        @test @inferred(size(A)) == (10,2)
        @test @inferred(A[5]) == @inferred(A[5,1]) == 5
        @test @inferred(A[11]) == @inferred(A[1,2]) == 2
        b = Array{Int}(undef, 11, 2)
        @test_throws DimensionMismatch copyto!(b, A)
        b = Array{Int}(undef, 10, 2)
        @test_broken @allocated(copyto!(b, A)) == 0
        @test @allocated(copyto!(b, A)) ≤ 200
        @test b == hcat(A.args...)
        @test copy(A) === A
        @test vec(A) == vec(Matrix(A))
        @test vec(A) === Vcat(1:10,2:11)
        @test A' == Matrix(A)'
        @test A' === Vcat((1:10)', (2:11)')

        A = Hcat(Vector(1:10), Vector(2:11))
        b = Array{Int}(undef, 10, 2)
        copyto!(b, A)
        @test b == hcat(A.args...)
        @test @allocated(copyto!(b, A)) == 0
        @test @allocated(copyto!(b, A)) ≤ 100
        @test copy(A) isa Hcat
        @test copy(A) == A
        @test copy(A) !== A
        @test vec(A) == vec(Matrix(A))
        @test vec(A) === Vcat(A.args...)
        @test A' == Matrix(A)'
        @test_throws BoundsError A[11,1] = 5
        @test_throws BoundsError A[5,3] = 5

        A = @inferred(Hcat(1, zeros(1,5)))
        @test A == hcat(1, zeros(1,5))
        @test vec(A) == vec(Matrix(A))
        @test A' == Matrix(A)'

        A = @inferred(Hcat(Vector(1:10), randn(10, 2)))
        b = Array{Float64}(undef, 10, 3)
        copyto!(b, A)
        @test b == hcat(A.args...)
        @test @allocated(copyto!(b, A)) == 0
        @test vec(A) == vec(Matrix(A))

        A = Hcat(randn(5).+im.*randn(5), randn(5,2).+im.*randn(5,2))
        b = Array{ComplexF64}(undef, 5, 3)
        copyto!(b, A)
        @test b == hcat(A.args...)
        @test @allocated(copyto!(b, A)) == 0
        @test vec(A) == vec(Matrix(A))
        @test A' == Matrix(A)'
        @test transpose(A) == transpose(Matrix(A))

        @testset "getindex bug" begin
            A = randn(3,3)
            H = Hcat(A,A)
            @test H[1,1] == applied(hcat,A,A)[1,1] == A[1,1]
        end

        @testset "adjoint vec / permutedims" begin
            @test vec(Hcat([1,2]', 3)) == 1:3
            @test permutedims(Hcat([1,2]', 3)) == reshape(1:3,3,1)
        end

        @testset "indexing" begin
            A = Hcat(randn(2,2), randn(2,3))
            @test A[2,1:5] == A[2,:] == Matrix(A)[2,:]
            @test A[1:2,2] == A[:,2] == Matrix(A)[:,2]
            @test A[:,2] == Matrix(A)[:,2]
            @test A[:,:] == A[1:2,:] == A[:,1:5] == A[1:2,1:5] == A
        end

        @testset "Hcat getindex" begin
            A = Hcat(1, (1:10)')
            @test A[1,:] isa Vcat{<:Any,1}
            @test A[1,:][1:10] == A[1,1:10]
        end
    end

    @testset "Hcat/Vcat adjoints" begin
        v = Vcat(1, 1:5)
        h = Hcat(1:5, 2:6)
        @test v' isa Adjoint
        @test transpose(v) isa Transpose
        @test MemoryLayout(v') isa DualLayout{ApplyLayout{typeof(hcat)}}
        @test MemoryLayout(transpose(v)) isa DualLayout{ApplyLayout{typeof(hcat)}}
        @test MemoryLayout(Adjoint(h)) isa ApplyLayout{typeof(vcat)}
        @test MemoryLayout(Transpose(h)) isa ApplyLayout{typeof(vcat)}
        @test copy(v') ≡ v'
        @test copy(Adjoint(h)) ≡ h'
        @test copy(transpose(v)) ≡ transpose(v)
        @test copy(Transpose(h)) ≡ transpose(h)

        @test arguments(v') ≡ (1, (1:5)')
        @test arguments(Adjoint(h)) ≡ ((1:5)', (2:6)')
        @test arguments(transpose(v)) ≡ (1, transpose(1:5))
        @test arguments(Transpose(h)) ≡ (transpose(1:5), transpose(2:6))

        @test v'*(1:6) ≡ 71

        A = Vcat(rand(2,3), randn(3,3))
        @test A' isa Hcat
        @test transpose(A) isa Hcat
        @test MemoryLayout(Adjoint(A)) isa ApplyLayout{typeof(hcat)}
        @test MemoryLayout(Transpose(A)) isa ApplyLayout{typeof(hcat)}
        @test Hcat(Adjoint(A)) == Hcat(Transpose(A)) == transpose(A) == A'
    end

    @testset "Hvcat" begin
        A = ApplyArray(hvcat, 2, randn(5,5), randn(5,6), randn(6,5), randn(6,6))
        @test eltype(A) == Float64
        @test A == hvcat(A.args...)
        @test_throws BoundsError A[2,12]
        @test_throws BoundsError A[12,2]
        @test A[[1,6],3] == Matrix(A)[[1,6],3]
        @test copyto!(similar(A), A) == A

        B = ApplyArray(hvcat, (3,2,1), randn(5,2), ones(5,3), randn(5,6), randn(6,5), randn(6,6), randn(2,11))
        @test eltype(B) == Float64
        @test B == hvcat(B.args...)
        @test_throws BoundsError A[2,14]
        @test_throws BoundsError A[14,2]
        @test copyto!(similar(B), B) == B

        A = ApplyArray(hvcat, 2, 1, 2, 3, 4)
        @test A == copyto!(similar(A), A) == [1 2; 3 4]
        @test_throws DimensionMismatch copyto!(zeros(Int,1,1), A)
        B = ApplyArray(hvcat, 3, 1, 2, 3, 4)
        @test_throws ArgumentError copyto!(similar(B), B)
        B = ApplyArray(hvcat, (1,2), 1, 2)
        @test_throws ArgumentError copyto!(zeros(Int,1,2), B)

        V = ApplyArray(hvcat, 2, [1,2], [3,4], [5,6], [7,8])
        @test V == [1 3; 2 4; 5 7; 6 8]

        W = ApplyArray(hvcat, 2, [1,2], [3], [5,6], [7,8])
        @test_throws ArgumentError copyto!(similar(W), W)
    end

    @testset "DefaultApplyStyle" begin
        v = Applied{DefaultApplyStyle}(vcat, (1, zeros(3)))
        @test v[1] == 1
        v = Applied{DefaultApplyStyle}(vcat, (1, zeros(3,1)))
        @test v[1,1] == 1
        H = Applied{DefaultApplyStyle}(hcat, (1, zeros(1,3)))
        @test H[1,1] == 1
    end

    @testset "setindex!" begin
        x = randn(5)
        y = randn(6)
        A = Vcat(x, y, 3)
        A[1] = 1
        @test A[1] == x[1] == 1
        A[6] = 2
        @test A[6] == y[1] == 2
        @test_throws MethodError A[12] = 3
        @test_throws BoundsError A[13] = 3

        x = randn(2,2); y = randn(3,2)
        A = Vcat(x,y)
        A[1,1] = 1
        @test A[1,1] == x[1,1] == 1
        A[3,1] = 2
        @test A[3,1] == y[1,1] == 2
        A[6] = 3
        @test A[1,2] == x[1,2] == 3

        x = randn(2,2); y = randn(2,3)
        B = Hcat(x,y)
        B[1,1] = 1
        @test B[1,1] == x[1,1] == 1
        B[1,3] = 2
        @test B[1,3] == y[1,1] == 2
    end

    @testset "fill!" begin
        A = Vcat([1,2,3],[4,5,6])
        fill!(A,2)
        @test A == fill(2,6)

        A = Vcat(2,[4,5,6])
        @test fill!(A,2) == fill(2,4)
        @test_throws ArgumentError fill!(A,3)

        A = Hcat([1,2,3],[4,5,6])
        fill!(A,2)
        @test A == fill(2,3,2)
    end

    @testset "Mul" begin
        A = Hcat([1.0 2.0],[3.0 4.0])
        B = Vcat([1.0,2.0],[3.0,4.0])

        @test MemoryLayout(typeof(A)) isa ApplyLayout{typeof(hcat)}
        @test MemoryLayout(typeof(B)) isa ApplyLayout{typeof(vcat)}
        @test A*B == Matrix(A)*Vector(B) == mul!(Vector{Float64}(undef,1),A,B) == (Vector{Float64}(undef,1) .= @~ A*B)
        @test materialize!(MulAdd(1.1,A,B,2.2,[5.0])) == 1.1*Matrix(A)*Vector(B)+2.2*[5.0]

        @test B * A ≈ Array(B) * Array(A) ≈ Vcat([1 2]', [3 4]') * A

        @test B * BroadcastArray(exp, [1 2]) ≈ B * exp.([1 2])

        A = Hcat([1.0 2.0; 3 4],[3.0 4.0; 5 6])
        B = Vcat([1.0,2.0],[3.0,4.0])
        @test MemoryLayout(typeof(A)) isa ApplyLayout{typeof(hcat)}
        @test MemoryLayout(typeof(B)) isa ApplyLayout{typeof(vcat)}
        @test A*B == Matrix(A)*Vector(B) == mul!(Vector{Float64}(undef,2),A,B) == (Vector{Float64}(undef,2) .= @~ A*B)
        @test materialize!(MulAdd(1.1,A,B,2.2,[5.0,6])) ≈ 1.1*Matrix(A)*Vector(B)+2.2*[5.0,6]

        A = Hcat([1.0 2.0; 3 4],[3.0 4.0; 5 6])
        B = Vcat([1.0 2.0; 3 4],[3.0 4.0; 5 6])
        @test MemoryLayout(typeof(A)) isa ApplyLayout{typeof(hcat)}
        @test MemoryLayout(typeof(B)) isa ApplyLayout{typeof(vcat)}
        @test A*B == Matrix(A)*Matrix(B) == mul!(Matrix{Float64}(undef,2,2),A,B) == (Matrix{Float64}(undef,2,2) .= @~ A*B)
        @test materialize!(MulAdd(1.1,A,B,2.2,[5.0 6; 7 8])) ≈ 1.1*Matrix(A)*Matrix(B)+2.2*[5.0 6; 7 8]
    end

    @testset "broadcast" begin
        x = Vcat(1:2, [1,1,1,1,1], 3)
        y = 1:8
        f = (x,y) -> cos(x*y)
        @test f.(x,y) isa Vcat
        @test @inferred(broadcast(f,x,y)) == f.(Vector(x), Vector(y))

        @test (x .+ y) isa Vcat
        @test (x .+ y).args[1] isa AbstractRange
        @test (x .+ y).args[end] isa Int

        z = Vcat(1:2, [1,1,1,1,1], 3)
        @test (x .+ z) isa BroadcastArray
        @test (x + z) isa BroadcastArray
        @test Vector( x .+ z) == Vector( x + z) == Vector(x) + Vector(z) == z .+ x

        @testset "Lazy mixed with Static treats as Lazy" begin
            s = SVector(1,2,3,4,5,6,7,8)
            @test f.(x , s) isa Vcat
            @test f.(x , s) == f.(Vector(x), Vector(s))
        end

        @testset "special cased" begin
            @test Vcat(1, Ones(5))  + Vcat(2, Fill(2.0,5)) ≡ Vcat(3, Fill(3.0,5))
            @test Vcat(SVector(1,2,3), Ones(5))  + Vcat(SVector(4,5,6), Fill(2.0,5)) ≡ Vcat(SVector(5,7,9), Fill(3.0,5))
            @test Vcat([1,2,3],Fill(1,7)) .* Zeros(10) ≡ Zeros(10) .* Vcat([1,2,3],Fill(1,7)) ≡ Zeros(10)
        end

        @testset "2-arg" begin
            a = Vcat([1,2], 1:3)
            b = Vcat([1], 1:4)
            @test a+b == b+a
        end

        H = Hcat(1, zeros(1,10))
        @test H/2 isa Hcat
        @test 2\H isa Hcat
        @test H./Ref(2) isa Hcat
        @test Ref(2).\H isa Hcat
        @test H/2  == H./Ref(2) == 2\H == Ref(2) .\ H == [1/2 zeros(1,10)]
    end

    @testset "norm" begin
        for a in (Vcat(1,2,Fill(5,3)), Hcat([1,2],randn(2,2)), Vcat(1,Float64[])),
            p in (-Inf, 0, 0.1, 1, 2, 3, Inf)
            @test norm(a,p) ≈ norm(Array(a),p)
        end
    end

    @testset "SubV/Hcat" begin
        A = Vcat(1,[2,3], Fill(5,10))
        V = view(A,3:5)
        @test MemoryLayout(V) isa ApplyLayout{typeof(vcat)}
        @inferred(arguments(V))
        @test arguments(V)[1] ≡ Fill(1,0)
        @test A[parentindices(V)...] == copy(V) == Array(A)[parentindices(V)...]

        A = Vcat((1:100)', Zeros(1,100),Fill(1,2,100))
        V = view(A,:,3:5)
        @test MemoryLayout(V) isa ApplyLayout{typeof(vcat)}
        @test A[parentindices(V)...] == copy(V) == Array(A)[parentindices(V)...]
        V = view(A,2:3,3:5)
        @test MemoryLayout(V) isa ApplyLayout{typeof(vcat)}
        @test A[parentindices(V)...] == copy(V) == Array(A)[parentindices(V)...]
    end

    @testset "col/rowsupport" begin
        H = Hcat(Diagonal([1,2,3]), Zeros(3,3), Diagonal([1,2,3]))
        V = Vcat(Diagonal([1,2,3]), Zeros(3,3), Diagonal([1,2,3]))
        @test colsupport(H,2) == rowsupport(V,2) == 2
        @test colsupport(H,4) == rowsupport(V,4) == 1:0
        @test colsupport(H,8) == rowsupport(V,8) == 2
        @test colsupport(H,10) == rowsupport(V,10)== 1:0
        @test rowsupport(H,1) == colsupport(V,1) == 1:7
        @test rowsupport(H,2) == colsupport(V,2) == 2:8
        @test colsupport(H,3:4) == rowsupport(V,3:4) == Base.OneTo(3)
        @test rowsupport(H,2:3) == colsupport(V,2:3) == 2:9
    end


    @testset "print" begin
        H = Hcat(Diagonal([1,2,3]), Zeros(3,3))
        V = Vcat(Diagonal([1,2,3]), Zeros(3,3))
        @test stringmime("text/plain", H) == "hcat(3×3 $Diagonal{$Int, Vector{$Int}}, 3×3 Zeros{Float64}):\n 1.0   ⋅    ⋅    ⋅    ⋅    ⋅ \n  ⋅   2.0   ⋅    ⋅    ⋅    ⋅ \n  ⋅    ⋅   3.0   ⋅    ⋅    ⋅ "
        @test stringmime("text/plain", V) == "vcat(3×3 $Diagonal{$Int, Vector{$Int}}, 3×3 Zeros{Float64}):\n 1.0   ⋅    ⋅ \n  ⋅   2.0   ⋅ \n  ⋅    ⋅   3.0\n  ⋅    ⋅    ⋅ \n  ⋅    ⋅    ⋅ \n  ⋅    ⋅    ⋅ "
        v = Vcat(1, Zeros(3))
        @test colsupport(v,1) == 1:1
        @test stringmime("text/plain", v) == "vcat($Int, 3-element Zeros{Float64}):\n 1.0\n  ⋅ \n  ⋅ \n  ⋅ "
        A = Vcat(Ones{Int}(1,3), Diagonal(1:3))
        @test stringmime("text/plain", A) == "vcat(1×3 Ones{$Int}, 3×3 $Diagonal{$Int, UnitRange{$Int}}):\n 1  1  1\n 1  ⋅  ⋅\n ⋅  2  ⋅\n ⋅  ⋅  3"
    end

    @testset "number-vec-vcat-broadcast" begin
        v = Vcat(1, 1:3)
        @test Fill.(v, 3) isa Vcat
        @test Fill.(v, 3) == Fill.(Vcat([1], 1:3), 3)
    end

    @testset "cumsum(Vcat(::Number, ::Fill))" begin
        v = Vcat(2, Fill(3,5))
        @test cumsum(v) isa StepRangeLen{Int}
        @test cumsum(v) == cumsum(collect(v)) == accumulate(+, v)
        @test accumulate(-, v) == accumulate(-, collect(v))
        v = Vcat(2, Zeros{Int}(5))
        @test cumsum(v) isa Fill{Int}
        @test cumsum(v) == cumsum(collect(v)) == accumulate(+, v)
        @test accumulate(-, v) == accumulate(-, collect(v))
        v = Vcat(2, Ones{Int}(5))
        @test cumsum(v) isa UnitRange{Int}
        @test cumsum(v) == cumsum(collect(v)) == accumulate(+, v)
        @test accumulate(-, v) == accumulate(-, collect(v))

        v = Vcat(2, Fill(3.0,5))
        @test cumsum(v) isa StepRangeLen{Float64}
        @test cumsum(v) == cumsum(collect(v)) == accumulate(+, v)
        @test accumulate(-, v) == accumulate(-, collect(v))
        v = Vcat(2, Zeros(5))
        @test cumsum(v) isa Fill{Float64}
        @test cumsum(v) == cumsum(collect(v)) == accumulate(+, v)
        @test accumulate(-, v) == accumulate(-, collect(v))
        v = Vcat(2, Ones(5))
        @test cumsum(v) isa StepRangeLen{Float64}
        @test cumsum(v) == cumsum(collect(v)) == accumulate(+, v)
        @test accumulate(-, v) == accumulate(-, collect(v))
    end

    @testset "empty vcat" begin
        v = ApplyArray(vcat)
        @test v isa AbstractVector{Any}
        @test stringmime("text/plain", v) == "vcat()"
    end

    @testset "matrix indexing" begin
        a = Vcat(2, Ones(5))
        h = Hcat([2,3], Ones(2,5))
        @test MemoryLayout(view(a, [1 2; 1 2])) isa LazyLayout
        @test MemoryLayout(view(h, [1 2; 1 2], 1)) isa LazyLayout
        @test h[[1 2; 1 2],:] == Matrix(h)[[1 2; 1 2],:]
    end

    @testset "transpose" begin
        a = Vcat(2, Ones(5))
        b = Vcat(2, Ones(5)) .+ im

        @test exp.(transpose(a)) isa Transpose{<:Any,<:Vcat}
        @test exp.(a') isa Adjoint{<:Any,<:Vcat}
        @test exp.(transpose(a)) == exp.(a') == exp.(a)'

        @test exp.(transpose(b)) isa Transpose{<:Any,<:Vcat}
        @test exp.(b') isa BroadcastArray
        @test exp.(transpose(b)) == transpose(exp.(b))
        @test exp.(b') == exp.(b)'
    end

    @testset "Applied hvcat" begin
        A,B,C,D = randn(5,5), randn(5,6), randn(6,5), randn(6,6)
        M = ApplyArray(hvcat, 2, A, B, C, D)
        A = Applied(hvcat, 2, A, B, C, D)
        @test A[1,2] == M[1,2]
    end

    @testset "1 arg Vcat/Hcat" begin
        @test Array(Hcat()) == Array{Any}(undef,0,0)
        @test rowsupport(Hcat(Vcat(Zeros(3,1))),1:2) == colsupport(Vcat(Hcat(Zeros(1,3))),1:2)
    end

    @testset "reverse Vcat" begin
        A = Vcat([1 2 3], [4 5 6])
        @test A[2:-1:1,1:-1:1] == [4; 1 ;;]
    end

    @testset "resizedata! for non-cached" begin
        A = @inferred(Vcat(1:10, 1:20))
        @test resizedata!(A, 3) ≡ A
    end
end

end # module
