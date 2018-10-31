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

        Base.:*(A::$Typ, B::$Typ) = LazyArrays.materialize(LazyArrays.Mul(A,B))
        Base.:*(A::$Typ, B::AbstractMatrix) = LazyArrays.materialize(LazyArrays.Mul(A,B))
        Base.:*(A::$Typ, B::AbstractVector) = LazyArrays.materialize(LazyArrays.Mul(A,B))
        Base.:*(A::AbstractMatrix, B::$Typ) = LazyArrays.materialize(LazyArrays.Mul(A,B))

        Base.:*(A::AbstractTriangular, B::$Typ) = LazyArrays.materialize(LazyArrays.Mul(A,B))
        Base.:*(A::$Typ, B::AbstractTriangular) = LazyArrays.materialize(LazyArrays.Mul(A,B))
    end
    for Mod in (:Adjoint, :Transpose, :Symmetric, :Hermitian)
        ret = quote
            $ret

            LinearAlgebra.mul!(dest::AbstractMatrix, A::$Typ, b::$Mod{<:Any,<:AbstractMatrix}) =
                copyto!(dest, LazyArrays.Mul(A,b))

            LinearAlgebra.mul!(dest::AbstractVector, A::$Mod{<:Any,<:$Typ}, b::AbstractVector) =
                copyto!(dest, LazyArrays.Mul(A,b))

            Base.:*(A::$Mod{<:Any,<:$Typ}, B::$Mod{<:Any,<:$Typ}) = LazyArrays.materialize(LazyArrays.Mul(A,B))
            Base.:*(A::$Mod{<:Any,<:$Typ}, B::AbstractMatrix) = LazyArrays.materialize(LazyArrays.Mul(A,B))
            Base.:*(A::AbstractMatrix, B::$Mod{<:Any,<:$Typ}) = LazyArrays.materialize(LazyArrays.Mul(A,B))
            Base.:*(A::$Mod{<:Any,<:$Typ}, B::AbstractVector) = LazyArrays.materialize(LazyArrays.Mul(A,B))

            Base.:*(A::$Mod{<:Any,<:$Typ}, B::$Typ) = LazyArrays.materialize(LazyArrays.Mul(A,B))
            Base.:*(A::$Typ, B::$Mod{<:Any,<:$Typ}) = LazyArrays.materialize(LazyArrays.Mul(A,B))

            Base.:*(A::AbstractTriangular, B::$Mod{<:Any,<:$Typ}) = LazyArrays.materialize(LazyArrays.Mul(A,B))
            Base.:*(A::$Mod{<:Any,<:$Typ}, B::AbstractTriangular) = LazyArrays.materialize(LazyArrays.Mul(A,B))
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
