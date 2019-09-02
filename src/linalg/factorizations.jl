struct QLayout <: MemoryLayout end

MemoryLayout(::Type{<:AbstractQ}) = QLayout()

mulapplystyle(::QLayout, _) = LmulStyle()
mulapplystyle(::QLayout, ::LazyLayout) = LazyArrayApplyStyle()
transposelayout(::QLayout) = QLayout()


# we need to special case AbstractQ as it allows non-compatiple multiplication
function check_mul_axes(A::AbstractQ, B, C...) 
    axes(A.factors, 1) == axes(B, 1) || axes(A.factors, 2) == axes(B, 1) ||  
        throw(DimensionMismatch("First axis of B, $(axes(B,1)) must match either axes of A, $(axes(A))"))
    check_mul_axes(B, C...)
end


function copyto!(dest::AbstractArray{T}, M::Lmul{QLayout}) where T
    A,B = M.A,M.B
    if size(dest,1) == size(B,1) 
        copyto!(dest, B)
    else
        copyto!(view(dest,1:size(B,1),:), B)
        fill!(@view(dest[size(B,1)+1:end,:]), zero(T))
    end
    materialize!(Lmul(A,dest))
end

function copyto!(dest::AbstractArray, M::Ldiv{QLayout})
    A,B = M.A,M.B
    copyto!(dest, B)
    materialize!(Ldiv(A,dest))
end

materialize!(M::Ldiv{QLayout}) = materialize!(Lmul(M.A',M.B))