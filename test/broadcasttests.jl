using LazyArrays, ArrayLayouts, LinearAlgebra, FillArrays, StaticArrays, Tracker, Base64, Test
import LazyArrays: BroadcastLayout, arguments, LazyArrayStyle, sub_materialize
import Base: broadcasted

@testset "Broadcasting" begin
    @testset "BroadcastArray" begin
        a = randn(6)
        b = BroadcastArray(exp, a)
        @test BroadcastArray(b) == BroadcastVector(b) == b == copyto!(similar(b), b)

        @test b ==  Vector(b) == exp.(a)
        @test b[2:5] isa Vector
        @test b[2:5] == exp.(a[2:5])

        @test exp.(b) isa BroadcastVector
        @test b .+ SVector(1,2,3,4,5,6) isa BroadcastVector
        @test SVector(1,2,3,4,5,6) .+ b isa BroadcastVector

        A = randn(6,6)
        B = BroadcastArray(exp, A)
        
        @test Matrix(B) == exp.(A)
        @test B[1] == exp(A[1,1])
        @test B[7] == exp(A[1,2])

        C = BroadcastArray(+, A, 2)
        @test C == A .+ 2
        D = BroadcastArray(+, A, C)
        @test D == A + C

        @test sum(B) ≈ sum(exp, A)
        @test sum(C) ≈ sum(A .+ 2)
        @test prod(B) ≈ prod(exp, A)
        @test prod(C) ≈ prod(A .+ 2)

        x = Vcat([3,4], [1,1,1,1,1], 1:3)
        @test x .+ (1:10) isa Vcat
        @test (1:10) .+ x isa Vcat
        @test x + (1:10) isa Vcat
        @test (1:10) + x isa Vcat
        @test x .+ (1:10) == (1:10) .+ x == (1:10) + x == x + (1:10) == Vector(x) + (1:10)

        @test exp.(x) isa Vcat
        @test exp.(x) == exp.(Vector(x))
        @test x .+ 2 isa Vcat
        @test (x .+ 2).args[end] ≡ x.args[end] .+ 2 ≡ 3:5
        @test x .* 2 isa Vcat
        @test 2 .+ x isa Vcat
        @test 2 .* x isa Vcat

        A = Vcat([[1 2; 3 4]], [[4 5; 6 7]])
        @test A .+ Ref(I) == Ref(I) .+ A == Vcat([[2 2; 3 5]], [[5 5; 6 8]])

        @test BroadcastArray(*,1.1,[1 2])[1] == 1.1

        B = BroadcastArray(*, Diagonal(randn(5)), randn(5,5))
        @test B == broadcast(*,B.args...)
        @test colsupport(B,1) == rowsupport(B,1) == 1:1
        @test colsupport(B,3) == rowsupport(B,3) == 3:3
        @test colsupport(B,5) == rowsupport(B,5) == 5:5
        B = BroadcastArray(*, Diagonal(randn(5)), 2)
        @test B == broadcast(*,B.args...)
        @test colsupport(B,1) == rowsupport(B,1) == 1:1
        @test colsupport(B,3) == rowsupport(B,3) == 3:3
        @test colsupport(B,5) == rowsupport(B,5) == 5:5
        B = BroadcastArray(*, Diagonal(randn(5)), randn(5))
        @test B == broadcast(*,B.args...)
        @test colsupport(B,1) == rowsupport(B,1) == 1:1
        @test colsupport(B,3) == rowsupport(B,3) == 3:3
        @test colsupport(B,5) == rowsupport(B,5) == 5:5

        B = BroadcastArray(+, Diagonal(randn(5)), 2)
        @test colsupport(B,1) == rowsupport(B,1) == 1:5
        @test colsupport(B,3) == rowsupport(B,3) == 1:5
        @test colsupport(B,5) == rowsupport(B,5) == 1:5

        @testset "different type" begin
            B = BroadcastArray{Float64}(+, [1,2,3], 2)
            @test eltype(B) == Float64
            @test B[1] ≡ 3.0
        end
    end

    @testset "vector*matrix broadcasting #27" begin
        H = [1., 0.]
        @test applied(*, H, H') .+ 1 == H*H' .+ 1
        B =  randn(2,2)
        @test applied(*, H, H') .+ B == H*H' .+ B
    end

    @testset "BroadcastArray +" begin
        a = BroadcastArray(+, randn(400), randn(400))
        b = similar(a)
        copyto!(b, a)
        @test @allocated(copyto!(b, a)) == 0
        @test b == a
    end

    @testset "Lazy range" begin
        @test broadcasted(LazyArrayStyle{1}(), +, 1:5) ≡ 1:5
        @test broadcasted(LazyArrayStyle{1}(), +, 1, 1:5) ≡ 2:6
        @test broadcasted(LazyArrayStyle{1}(), +, 1:5, 1) ≡ 2:6

        @test broadcasted(LazyArrayStyle{1}(), +, Fill(2,5)) ≡ Fill(2,5)
        @test broadcasted(LazyArrayStyle{1}(), +, 1, Fill(2,5)) ≡ Fill(3,5)
        @test broadcasted(LazyArrayStyle{1}(), +, Fill(2,5), 1) ≡ Fill(3,5)
        @test broadcasted(LazyArrayStyle{1}(), +, Ref(1), Fill(2,5)) ≡ Fill(3,5)
        @test broadcasted(LazyArrayStyle{1}(), +, Fill(2,5), Ref(1)) ≡ Fill(3,5)
        @test broadcasted(LazyArrayStyle{1}(), +, 1, Fill(2,5)) ≡ Fill(3,5)
        @test broadcasted(LazyArrayStyle{1}(), +, Fill(2,5), Fill(3,5)) ≡ Fill(5,5)

        @test broadcasted(LazyArrayStyle{1}(), *, Zeros(5), Zeros(5)) ≡ Zeros(5)
        @test broadcasted(LazyArrayStyle{1}(), *, Fill(2,5), 1:5) ≡ broadcasted(LazyArrayStyle{1}(), *, 1:5, Fill(2,5)) ≡ 2 * (1:5)
        @test broadcasted(LazyArrayStyle{1}(), *, Zeros(5), 1:5) ≡ broadcasted(LazyArrayStyle{1}(), *, 1:5, Zeros(5)) ≡ Zeros(5)
        @test broadcasted(LazyArrayStyle{1}(), *, Ones{Int}(5), 1:5) ≡ broadcasted(LazyArrayStyle{1}(), *, 1:5, Ones{Int}(5)) ≡ 1:5

        b = BroadcastArray(exp, randn(5))
        @test b .* Zeros(5) ≡ Zeros(5)
        @test Zeros(5) .* b ≡ Zeros(5)
    end

    @testset "Sub-broadcast" begin
        A = BroadcastArray(exp,randn(5,5))
        V = view(A, 1:2,2:3)
        @test MemoryLayout(typeof(V)) isa BroadcastLayout{typeof(exp)}
        @test BroadcastArray(V) == A[1:2,2:3] == Array(A)[1:2,2:3]

        B = BroadcastArray(-, randn(5,5), randn(5))
        V = view(B, 1:2,2:3)
        @test MemoryLayout(typeof(V)) isa BroadcastLayout{typeof(-)}
        @test BroadcastArray(V) == B[1:2,2:3] == Array(B)[1:2,2:3]
    end

    @testset "AdjTrans" begin
        A = BroadcastArray(exp,randn(5,5))
        @test MemoryLayout(typeof(transpose(A))) isa BroadcastLayout{typeof(exp)}
        @test MemoryLayout(typeof(A')) isa BroadcastLayout{typeof(exp)}
        @test BroadcastArray(A') == BroadcastArray(transpose(A)) == A' == Array(A)'

        B = BroadcastArray(-, randn(5,5), randn(5))
        @test MemoryLayout(typeof(transpose(B))) isa BroadcastLayout{typeof(-)}
        @test MemoryLayout(typeof(B')) isa BroadcastLayout{typeof(-)}  
        @test BroadcastArray(B') == BroadcastArray(transpose(B)) == B' == Array(B)'      

        Vc = view(B', 1:2,1:3)
        Vt = view(transpose(B), 1:2,1:3)
        @test MemoryLayout(typeof(Vc)) isa BroadcastLayout{typeof(-)}
        @test MemoryLayout(typeof(Vt)) isa BroadcastLayout{typeof(-)}
        @test arguments(Vc) == (B.args[1][1:3,1:2]', permutedims(B.args[2][1:3]))
        @test arguments(Vt) == (transpose(B.args[1][1:3,1:2]), permutedims(B.args[2][1:3]))
        @test BroadcastArray(Vc) == BroadcastArray(Vt) == Vc == Vt == (Array(B)')[1:2,1:3]
        
        Vc = view(B,1:3,1:2)'
        Vt = transpose(view(B,1:3,1:2))
        @test MemoryLayout(typeof(Vc)) isa BroadcastLayout{typeof(-)}
        @test MemoryLayout(typeof(Vt)) isa BroadcastLayout{typeof(-)}
        @test arguments(Vc) == (B.args[1][1:3,1:2]', permutedims(B.args[2][1:3]))
        @test arguments(Vt) == (transpose(B.args[1][1:3,1:2]), permutedims(B.args[2][1:3]))
        @test BroadcastArray(Vc) == BroadcastArray(Vt) == Vc == (Array(B)')[1:2,1:3]      
    end

    @testset "copy" begin
        a = LazyArray(broadcasted(+, param(rand(3, 3)), 1))
        @test @inferred(copy(a)) isa BroadcastArray{<:Tracker.TrackedReal}

        a = randn(5)
        A = BroadcastArray(*, 2, a)
        @test copy(A) ≡ map(copy,A) ≡ A
        @test copy(A') ≡ A'
    end

    @testset "Number .* A" begin
        a = randn(5)
        A = BroadcastArray(*, 2, a)
        V = view(A,1:3)
        @test arguments(V) == (2,a[1:3])
        @test BroadcastArray(V) == V == 2a[1:3]
    end

    @testset "vec .* A" begin
        A = BroadcastArray(*, 1:3, randn(3,4))
        Ã = BroadcastArray(*, A.args[2],  1:3)
        B = randn(4,2)
        C = BroadcastArray(*, randn(4,2), randn(4,2))
        b = randn(2)

        @test A*B ≈ Matrix(A)*B
        @test Ã*B ≈ Matrix(Ã)*B
        @test A*C ≈ Matrix(A)*Matrix(C)
        @test Ã*C ≈ Matrix(Ã)*Matrix(C)
        @test A[:,2] ≈ Ã[:,2] ≈ Matrix(A)[:,2]
        @test C*b ≈ Matrix(C)*b

        D = Diagonal(Fill(2,4))
        @test A*D ≈ Matrix(A)*D
    end

    @testset "broadcasted which simplifies" begin
        a = BroadcastVector{Float64}(*, Zeros(10), randn(10))
        @test @inferred(a[1]) == 0.0
        @test @inferred(a[1:5]) ≡ Zeros(5)
        @test @inferred(a[[1,2,4]]) ≡ Zeros(3)
        @test broadcasted(a) ≡ Zeros(10)
        @test_throws TypeError Base.Broadcast.Broadcasted(a)
        @test broadcasted(view(a,1:3)) ≡ Zeros(3)
    end

    @testset "array-valued Broadcast" begin
        a = BroadcastArray(*, 1:3, [[1,2],[3,4],[5,6]])
        @test a == broadcast(*, 1:3, [[1,2],[3,4],[5,6]])
        @test a[2] == [6,8]
        @test a[1:2] == [[1,2], [6,8]]     
    end

    @testset "submaterialize" begin
        a = BroadcastArray(/, randn(1000), 2)
        @test a[3:10] ≈ a.args[1][3:10]/2
        @test MemoryLayout(a') isa DualLayout{BroadcastLayout{typeof(/)}}
        @test (a')[:,3:10] isa Adjoint
        @test (a')[:,3:10] ≈ a[3:10]'

        @test BroadcastArray(view(a',1,3:10)) == a[3:10]
    end

    @testset "adjoint broadcast" begin
        a = BroadcastArray(exp, 1:5)
        b = randn(5)
        @test MemoryLayout(a') isa DualLayout{BroadcastLayout{typeof(exp)}}
        @test a'b ≈ Vector(a)'b
        @test BroadcastArray(a')b ≈ [a'b]
    end

    @testset "show" begin
        x = 1:3    
        @test stringmime("text/plain", BroadcastArray(factorial, 1:3)) == "factorial.(3-element UnitRange{$Int}):\n 1\n 2\n 6"
        @test stringmime("text/plain", BroadcastArray(^, 1:3, 2)) == "(3-element UnitRange{$Int}) .^ $Int:\n 1\n 4\n 9"
        @test stringmime("text/plain", BroadcastArray(@~ x .^ 2)) == "(3-element UnitRange{$Int}) .^ 2:\n 1\n 4\n 9"
        @test stringmime("text/plain", BroadcastArray(@~ x .^ 2)') == "((3-element UnitRange{$Int}) .^ 2)':\n 1  4  9"
        @test stringmime("text/plain", transpose(BroadcastArray(@~ x .^ 2))) == "transpose((3-element UnitRange{$Int}) .^ 2):\n 1  4  9"

        @test stringmime("text/plain", BroadcastArray(+, [1,2], 2)) == "(2-element Vector{$Int}) .+ ($Int):\n 3\n 4"
        @test stringmime("text/plain", BroadcastArray(+, [1,2])) == "(+).(2-element Vector{$Int}):\n 1\n 2"
        @test stringmime("text/plain", BroadcastArray(+, [1,2], 2)) == "(2-element Vector{$Int}) .+ ($Int):\n 3\n 4"
        @test stringmime("text/plain", BroadcastArray(mod, [1,2], 2)) == "mod.(2-element Vector{$Int}, $Int):\n 1\n 0"
    end

    @testset "offset indexing" begin
        v = BroadcastArray(+, SubArray(1:3, (Base.IdentityUnitRange(1:3),)), 1)
        @test axes(v) == (Base.IdentityUnitRange(1:3),)
        @test v[1] == 2
        if VERSION < v"1.6.2"
            @test stringmime("text/plain", v) == "(3-element view(::UnitRange{$Int}, :) with eltype $Int with indices 1:3) .+ ($Int) with indices 1:3:\n 2\n 3\n 4"
            @test stringmime("text/plain", v') == "((3-element view(::UnitRange{$Int}, :) with eltype $Int with indices 1:3) .+ ($Int) with indices 1:3)' with indices Base.OneTo(1)×1:3:\n 2  3  4"
            @test stringmime("text/plain", transpose(v)) == "transpose((3-element view(::UnitRange{$Int}, :) with eltype $Int with indices 1:3) .+ ($Int) with indices 1:3) with indices Base.OneTo(1)×1:3:\n 2  3  4"
        else
            @test stringmime("text/plain", v) == "(3-element view(::UnitRange{$Int}, Base.IdentityUnitRange(1:3)) with eltype $Int with indices 1:3) .+ ($Int) with indices 1:3:\n 2\n 3\n 4"
            @test stringmime("text/plain", v') == "((3-element view(::UnitRange{$Int}, Base.IdentityUnitRange(1:3)) with eltype $Int with indices 1:3) .+ ($Int) with indices 1:3)' with indices Base.OneTo(1)×1:3:\n 2  3  4"
            @test stringmime("text/plain", transpose(v)) == "transpose((3-element view(::UnitRange{$Int}, Base.IdentityUnitRange(1:3)) with eltype $Int with indices 1:3) .+ ($Int) with indices 1:3) with indices Base.OneTo(1)×1:3:\n 2  3  4"
        end
    end

    @testset "Ref" begin
        A = BroadcastArray(norm, Ref([1,2]), [1,2])
        @test A == [norm([1,2],1), norm([1,2],2)]
        Ac = BroadcastArray(A')
        At = BroadcastArray(transpose(A))
        @test Ac == At == [norm([1,2],1) norm([1,2],2)] 
    end

    @testset "large args tuple_type_memorylayouts" begin
        a = randn(5)
        @test MemoryLayout(BroadcastArray(+, a)) isa BroadcastLayout{typeof(+)}
        @test MemoryLayout(BroadcastArray(+, a, a)) isa BroadcastLayout{typeof(+)}
        @test MemoryLayout(BroadcastArray(+, a, a, a)) isa BroadcastLayout{typeof(+)}
        @test MemoryLayout(BroadcastArray(+, a, a, a, a)) isa BroadcastLayout{typeof(+)}
        @test MemoryLayout(BroadcastArray(+, a, a, a, a, a)) isa BroadcastLayout{typeof(+)}
        @test MemoryLayout(BroadcastArray(+, a, a, a, a, a, a)) isa BroadcastLayout{typeof(+)}
    end

    @testset "block array axes broadcasting" begin
        # Special cases to support non-allocating block sizes 1:N as in 2D polynomials
        n = BroadcastArray(Fill, Base.OneTo(5), Base.OneTo(5))
        k = BroadcastArray(Base.OneTo,Base.OneTo(5))
        z = BroadcastArray(Zeros, Base.OneTo(5))
        v = BroadcastArray(Vcat, n, k)
        @test map(length, n) ≡ map(length, k) ≡ map(length, z) ≡ Base.OneTo(5)
        @test map(length, v) ≡ 2 * (1:5)
        @test map(length, BroadcastArray(Fill,Base.OneTo(5))) == ones(5)

        @test broadcast(length, n) ≡ broadcast(length, k) ≡ Base.OneTo(5)

        @test map(length, BroadcastArray(Fill, Base.OneTo(5), 2)) ≡ Fill(2, 5)
        @test map(length, BroadcastArray(Fill, 2, Base.OneTo(5))) == length.(Fill.(2, Base.OneTo(5)))
    end

    @testset "sub_materialize of row slice" begin
        x = [0.1,0.2]
        B = BroadcastArray(*, x, (1:10)')
        @test sub_materialize(view(B,2,2:3)) == B[2,2:3] == 0.2 * (2:3)
        @test sub_materialize(view(B,[1,2],2)) == B[[1,2],2] == x * 2
        @test sub_materialize(view(B,[1,2],2:3)) == B[[1,2],2:3] == x * (2:3)'
        @test arguments(view(B, 2,3)) == (0.2,3)
        
        C = BroadcastArray(*, 1:10, x')
        @test sub_materialize(view(C,2,1:2)) == C[2,1:2] == 2 * x
        @test sub_materialize(view(C,[1,2],2)) == C[[1,2],2] == x[2] * (1:2)
        @test sub_materialize(view(C,[1,2],1:2)) == C[[1,2],1:2] == x' .* (1:2)
        @test arguments(view(C, 3,2)) == (3,0.2)
    end

    @testset "quasi-broadcasting" begin
        # test support needed in QuasiArrays.jl
        struct Inclusion end
        Base.axes(x::Inclusion) = (x,)
        Base.getindex(::Inclusion, y) = y
        Base.view(::Inclusion, y) = y

        @test LazyArrays.__broadcastview((0.1,2:3), Inclusion(),(1:5)') == (0.1, 2:3)
        @test LazyArrays.__broadcastview(([0.1,0.2],2:3), Inclusion(),(1:5)') == ([0.1,0.2], [2 3])
        # TODO: special case adjtrans so it becomes `2` instead of `[2]`
        @test LazyArrays.__broadcastview(([0.1,0.2],2), Inclusion(),(1:5)') == ([0.1,0.2], [2])
    end

    @testset "UniformScaling arthmetic" begin
        A = BroadcastArray(*,randn(5),randn(5,5))
        @test A + I == I + A
    end
end