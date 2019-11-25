struct Interlace{T, N, AA, INDS} <: AbstractArray{T,N}
    arrays::AA
    inds::INDS
end


_sortunion(inds...) = sort!(union(inds...))
function _sortunion(inds::Vararg{StepRange,N}) where N
    all(isequal(N), map(step, inds)) || throw(ArgumentError("incompatible"))
    sort([map(first, inds)...]) == OneTo(N) || throw(ArgumentError("incompatible"))
    n = mapreduce(length, +, inds)
    maximum(map(last, inds)) == n || throw(ArgumentError("incompatible lengths"))
    OneTo(n)
end


function check_interlace_inds(a, inds)
    map(length,a) == map(length,inds) || throw(ArgumentError("Lengths must be compatible"))
    n = mapreduce(length, +, a)
    _sortunion(inds...) == OneTo(n) ||  throw(ArgumentError("Every index must be mapped to"))
end

function Interlace(a::NTuple{M,AbstractVector{T}}, inds::NTuple{M,AbstractVector{Int}}) where {T,M} 
    check_interlace_inds(a, inds)
    Interlace{T,1,typeof(a), typeof(inds)}(a, inds)
end

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

function copyto!(dest::AbstractVector, src::Interlace{<:Any,1})
    for (a,ind) in zip(src.arrays, src.inds)
        copyto!(view(dest, ind), a)
    end
    dest 
end

Interlace(a::AbstractVector, b::AbstractVector) = 
    Interlace((a,b), (1:2:(2length(a)-1), 2:2:2length(b)))

interlace(a...) = Array(Interlace(a...))    