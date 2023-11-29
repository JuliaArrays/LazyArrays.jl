# Infinite Arrays implementation from
# https://github.com/JuliaLang/julia/blob/master/test/testhelpers/InfiniteArrays.jl
module InfiniteArrays
    export OneToInf, Infinity

    """
       Infinity()
    Represents infinite cardinality. Note that `Infinity <: Integer` to support
    being treated as an index.
    """
    struct Infinity <: Integer end

    Base.:(==)(::Infinity, ::Int) = false
    Base.:(==)(::Int, ::Infinity) = false
    Base.:(<)(::Int, ::Infinity) = true
    Base.:(<)(::Infinity, ::Int) = false
    Base.:(≤)(::Int, ::Infinity) = true
    Base.:(≤)(::Infinity, ::Int) = false
    Base.:(≤)(::Infinity, ::Infinity) = true
    Base.:(-)(::Infinity, ::Int) = Infinity()
    Base.:(+)(::Infinity, ::Int) = Infinity()
    Base.:(:)(::Infinity, ::Infinity) = 1:0

    Base.:(+)(::Integer, ::Infinity) = Infinity()
    Base.:(+)(::Infinity, ::Integer) = Infinity()
    Base.:(*)(::Integer, ::Infinity) = Infinity()
    Base.:(*)(::Infinity, ::Integer) = Infinity()

    Base.isinf(::Infinity) = true

    abstract type AbstractInfUnitRange{T<:Real} <: AbstractUnitRange{T} end
    Base.length(r::AbstractInfUnitRange) = Infinity()
    Base.size(r::AbstractInfUnitRange) = (Infinity(),)
    Base.unitrange(r::AbstractInfUnitRange) = InfUnitRange(r)
    Base.last(r::AbstractInfUnitRange) = Infinity()
    Base.axes(r::AbstractInfUnitRange) = (OneToInf(),)

    Base.IteratorSize(::Type{<:AbstractInfUnitRange}) = Base.IsInfinite()

    """
        OneToInf(n)
    Define an `AbstractInfUnitRange` that behaves like `1:∞`, with the added
    distinction that the limits are guaranteed (by the type system) to
    be 1 and ∞.
    """
    struct OneToInf{T<:Integer} <: AbstractInfUnitRange{T} end

    OneToInf() = OneToInf{Int}()

    Base.axes(r::OneToInf) = (r,)
    Base.first(r::OneToInf{T}) where {T} = oneunit(T)
    Base.oneto(::Infinity) = OneToInf()

    struct InfUnitRange{T<:Real} <: AbstractInfUnitRange{T}
        start::T
    end
    Base.first(r::InfUnitRange) = r.start
    InfUnitRange(a::InfUnitRange) = a
    InfUnitRange{T}(a::AbstractInfUnitRange) where T<:Real = InfUnitRange{T}(first(a))
    InfUnitRange(a::AbstractInfUnitRange{T}) where T<:Real = InfUnitRange{T}(first(a))
    unitrange(a::AbstractInfUnitRange) = InfUnitRange(a)
    Base.:(:)(start::T, stop::Infinity) where {T<:Integer} = InfUnitRange{T}(start)
    function getindex(v::InfUnitRange{T}, i::Integer) where T
        @boundscheck i > 0 || Base.throw_boundserror(v, i)
        convert(T, first(v) + i - 1)
    end
end
