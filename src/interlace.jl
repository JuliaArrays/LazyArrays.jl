struct Interlace{T, N, AA, INDS} <: AbstractArray{T,N}
    arrays::AA
    inds::INDS
end

Interlace(a::AbstractArray{<:AbstractArray{T,N},N}, inds) where {T,N} = Interlace{T,N,typeof(a), typeof(inds)}(a, inds)

length(A::Interlace) = sum(length(A.arrays))
size(A::Interlace, m) = sum(size.(A.arrays,m))
size(A::Interlace{<:Any,1}) = (size(A,1),)
function getindex(A::Interlace{<:Any,1}, k::Integer)
    for (a,ind) in zip(A.arrays, A.inds)
        κ = findfirst(isequal(k), ind)
        isnothing(κ) || return a[something(κ)]
    end
    throw(BoundsError(A, k))
end
