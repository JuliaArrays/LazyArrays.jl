struct InterlaceLayout end

reshapedlayout(::ApplyLayout{typeof(vcat)}, _) = InterlaceLayout()
function arguments(::InterlaceLayout, A::ReshapedArray)
    args = arguments(ApplyLayout{typeof(Vcat)}(), parent(A))
    map(_permutedims, args)
end

function _copyto!(_, ::InterlaceLayout, dest::AbstractVector, A::AbstractVector)
    args = arguments(InterlaceLayout(), A)
    _interlace_copyto!(1, length(args), dest, args...)
end

function _interlace_copyto!(k, st, dest, a, b...)
    copyto!(view(dest, k:st:length(dest)), a)
    _interlace_copyto!(k+1, st, dest, b...)
end

_interlace_copyto!(k, st, dest) = dest