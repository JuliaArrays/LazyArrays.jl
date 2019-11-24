struct Interlace{T, N, AA, INDS} <: AbstractArray{T,N}
    arrays::AA
    inds::INDS
end

Interlace(a::NTuple{M,AbstractVector{T}}, inds::NTuple{M,AbstractVector{Int}}) where {T,M} = 
    Interlace{T,1,typeof(a), typeof(inds)}(a, inds)

length(A::Interlace) = sum(map(length,A.arrays))
size(A::Interlace, m) = sum(size.(A.arrays,m))
size(A::Interlace{<:Any,1}) = (size(A,1),)
function getindex(A::Interlace{<:Any,1}, k::Integer)
    for (a,ind) in zip(A.arrays, A.inds)
        κ = findfirst(isequal(k), ind)
        isnothing(κ) || return a[something(κ)]
    end
    throw(BoundsError(A, k))
end

Interlace(a::AbstractVector, b::AbstractVector) = 
    Interlace((a,b), (1:2:(2length(a)-1), 2:2:2length(b)))

interlace(a...) = Array(Interlace(a...))    