####
# This macro overrides mul! to call lazymul!
####


# support mul! by calling lazy mul
macro lazymul(Typ)
    ret = quote
        LinearAlgebra.mul!(dest::AbstractVector, A::$Typ, b::AbstractVector) =
            copyto!(dest, LazyArrays.Mul(A,b))

        LinearAlgebra.mul!(dest::AbstractMatrix, A::$Typ, b::AbstractMatrix) =
            copyto!(dest, LazyArrays.Mul(A,b))
        LinearAlgebra.mul!(dest::AbstractMatrix, A::$Typ, b::$Typ) =
            copyto!(dest, LazyArrays.Mul(A,b))

        Base.:*(A::$Typ, B::$Typ, C...) = LazyArrays.materialize(LazyArrays.Mul(A,B,C...))
        Base.:*(A::$Typ, B::AbstractMatrix, C...) = LazyArrays.materialize(LazyArrays.Mul(A,B,C...))
        Base.:*(A::$Typ, B::AbstractVector) = LazyArrays.materialize(LazyArrays.Mul(A,B))
        Base.:*(A::AbstractMatrix, B::$Typ, C...) = LazyArrays.materialize(LazyArrays.Mul(A,B,C...))
    end
    for Struc in (:AbstractTriangular, :Diagonal)
        ret = quote
            $ret

            Base.:*(A::$Struc, B::$Typ, C...) = LazyArrays.materialize(LazyArrays.Mul(A,B, C...))
            Base.:*(A::$Typ, B::$Struc, C...) = LazyArrays.materialize(LazyArrays.Mul(A,B, C...))
        end
    end
    for Mod in (:Adjoint, :Transpose, :Symmetric, :Hermitian)
        ret = quote
            $ret

            LinearAlgebra.mul!(dest::AbstractMatrix, A::$Typ, b::$Mod{<:Any,<:AbstractMatrix}) =
                copyto!(dest, LazyArrays.Mul(A,b))

            LinearAlgebra.mul!(dest::AbstractVector, A::$Mod{<:Any,<:$Typ}, b::AbstractVector) =
                copyto!(dest, LazyArrays.Mul(A,b))

            Base.:*(A::$Mod{<:Any,<:$Typ}, B::$Mod{<:Any,<:$Typ}, C...) = LazyArrays.materialize(LazyArrays.Mul(A,B, C...))
            Base.:*(A::$Mod{<:Any,<:$Typ}, B::AbstractMatrix, C...) = LazyArrays.materialize(LazyArrays.Mul(A,B, C...))
            Base.:*(A::AbstractMatrix, B::$Mod{<:Any,<:$Typ}, C...) = LazyArrays.materialize(LazyArrays.Mul(A,B, C...))
            Base.:*(A::$Mod{<:Any,<:$Typ}, B::AbstractVector, C...) = LazyArrays.materialize(LazyArrays.Mul(A,B, C...))

            Base.:*(A::$Mod{<:Any,<:$Typ}, B::$Typ, C...) = LazyArrays.materialize(LazyArrays.Mul(A,B, C...))
            Base.:*(A::$Typ, B::$Mod{<:Any,<:$Typ}, C...) = LazyArrays.materialize(LazyArrays.Mul(A,B, C...))

            Base.:*(A::$Mod{<:Any,<:$Typ}, B::Diagonal, C...) = LazyArrays.materialize(LazyArrays.Mul(A,B, C...))
            Base.:*(A::Diagonal, B::$Mod{<:Any,<:$Typ}, C...) = LazyArrays.materialize(LazyArrays.Mul(A,B, C...))

            Base.:*(A::AbstractTriangular, B::$Mod{<:Any,<:$Typ}, C...) = LazyArrays.materialize(LazyArrays.Mul(A,B, C...))
            Base.:*(A::$Mod{<:Any,<:$Typ}, B::AbstractTriangular, C...) = LazyArrays.materialize(LazyArrays.Mul(A,B, C...))
        end
    end

    esc(ret)
end

macro lazylmul(Typ)
    esc(quote
        LinearAlgebra.lmul!(A::$Typ, x::AbstractVector) = copyto!(x, LazyArrays.Mul(A,x))
        LinearAlgebra.lmul!(A::$Typ, x::AbstractMatrix) = copyto!(x, LazyArrays.Mul(A,x))
        LinearAlgebra.lmul!(A::$Typ, x::StridedVector) = copyto!(x, LazyArrays.Mul(A,x))
        LinearAlgebra.lmul!(A::$Typ, x::StridedMatrix) = copyto!(x, LazyArrays.Mul(A,x))
    end)
end


@lazymul AddMatrix
