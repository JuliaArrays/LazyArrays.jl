####
# This macro overrides mul! to call lazymul!
####


# support mul! by calling lazy mul
macro lazymul(Typ)
    esc(quote
        LinearAlgebra.mul!(dest::AbstractVector, A::$Typ, b::AbstractVector) =
            copyto!(dest, Mul(A,b))

        LinearAlgebra.mul!(dest::AbstractMatrix, A::$Typ, b::AbstractMatrix) =
            copyto!(dest, Mul(A,b))
        LinearAlgebra.mul!(dest::AbstractMatrix, A::$Typ, b::$Typ) =
            copyto!(dest, Mul(A,b))
        LinearAlgebra.mul!(dest::AbstractMatrix, A::$Typ, b::Adjoint{<:Any,<:AbstractMatrix}) =
            copyto!(dest, Mul(A,b))

        LinearAlgebra.mul!(dest::AbstractVector, A::Adjoint{<:Any,<:$Typ}, b::AbstractVector) =
            copyto!(dest, Mul(A,b))
        LinearAlgebra.mul!(dest::AbstractVector, A::Transpose{<:Any,<:$Typ}, b::AbstractVector) =
            copyto!(dest, Mul(A,b))

        LinearAlgebra.mul!(dest::AbstractVector, A::Symmetric{<:Any,<:$Typ}, b::AbstractVector) =
            copyto!(dest, Mul(A,b))
        LinearAlgebra.mul!(dest::AbstractVector, A::Hermitian{<:Any,<:$Typ}, b::AbstractVector) =
            copyto!(dest, Mul(A,b))
    end)
end

macro lazylmul(Typ)
    esc(quote
        LinearAlgebra.lmul!(A::$Typ, x::AbstractVector) = copyto!(x, Mul(A,x))
        LinearAlgebra.lmul!(A::$Typ, x::AbstractMatrix) = copyto!(x, Mul(A,x))
        LinearAlgebra.lmul!(A::$Typ, x::StridedVector) = copyto!(x, Mul(A,x))
        LinearAlgebra.lmul!(A::$Typ, x::StridedMatrix) = copyto!(x, Mul(A,x))
    end)
end
