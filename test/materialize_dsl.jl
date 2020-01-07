struct MyOperator{T}
    n::Int
    kind::Symbol
end

Base.axes(O::MyOperator) = (Base.OneTo(O.n),Base.OneTo(O.n))
Base.axes(O::MyOperator,i) = axes(O)[i]
Base.size(O::MyOperator) = (O.n,O.n)
Base.eltype(::MyOperator{T}) where T = T

struct MyApplyStyle <: ApplyStyle end

@materialize function *(Ac::Adjoint{<:Any,<:AbstractMatrix},
                        O::MyOperator,
                        B::AbstractMatrix)
    MyApplyStyle
    T -> begin
        A = parent(Ac)

        if O.kind == :diagonal
            Diagonal(Vector{T}(undef, O.n))
        else
            Tridiagonal(Vector{T}(undef, O.n-1),
                        Vector{T}(undef, O.n),
                        Vector{T}(undef, O.n-1))
        end
    end
    dest::Diagonal -> begin
        dest.diag .= 1
    end
    dest::Tridiagonal{T} -> begin
        dest.dl .= -2
        dest.d .= 1
        dest.du .= 3
    end
end

@testset "Materialize DSL" begin
    o = ones(10)
    M = ones(10,10)
    D = MyOperator{Float64}(10, :diagonal)
    T = MyOperator{ComplexF64}(10, :tridiagonal)

    @test LazyArrays.ApplyStyle(*, typeof(M'), typeof(D), typeof(M)) == MyApplyStyle()
    @test LazyArrays.ApplyStyle(*, typeof(M'), typeof(T), typeof(M)) == MyApplyStyle()

    d = apply(*, M', D, M)
    @test d isa Diagonal{Float64}
    @test all(d.diag .== 1)

    t = apply(*, M', T, M)
    @test t isa Tridiagonal
    @test all(t.dl .== -2)
    @test all(t.d .== 1)
    @test all(t.du .== 3)

    M̃ = ones(11,11)
    @test_throws DimensionMismatch apply(*, M̃', D, M̃)
end
