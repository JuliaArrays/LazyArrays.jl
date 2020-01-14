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

        Base.:*(A::$Typ, B::$Typ, C...) = LazyArrays.apply(*,A,B,C...)
        Base.:*(A::$Typ, B::AbstractMatrix, C...) = LazyArrays.apply(*,A,B,C...)
        Base.:*(A::$Typ, B::AbstractVector) = LazyArrays.apply(*,A,B)
        Base.:*(A::$Typ, B::LazyArrays.LazyVector) = LazyArrays.apply(*,A,B)
        Base.:*(A::AbstractMatrix, B::$Typ, C...) = LazyArrays.apply(*,A,B,C...)
        Base.:*(A::LinearAlgebra.AdjointAbsVec, B::$Typ, C...) = LazyArrays.apply(*,A,B,C...)
        Base.:*(A::LinearAlgebra.TransposeAbsVec, B::$Typ, C...) = LazyArrays.apply(*,A,B,C...)
    end
    if Typ ≠ :LazyMatrix
        ret = quote
            $ret
            Base.:*(A::$Typ, B::LazyArrays.LazyMatrix, C...) = LazyArrays.apply(*,A,B,C...)
            Base.:*(A::LazyArrays.LazyMatrix, B::$Typ, C...) = LazyArrays.apply(*,A,B,C...)
        end
    end
    for Struc in (:AbstractTriangular, :Diagonal)
        ret = quote
            $ret

            Base.:*(A::LinearAlgebra.$Struc, B::$Typ, C...) = LazyArrays.apply(*,A,B, C...)
            Base.:*(A::$Typ, B::LinearAlgebra.$Struc, C...) = LazyArrays.apply(*,A,B, C...)
        end
    end
    for Mod in (:Adjoint, :Transpose, :Symmetric, :Hermitian)
        ret = quote
            $ret

            LinearAlgebra.mul!(dest::AbstractMatrix, A::$Typ, b::$Mod{<:Any,<:AbstractMatrix}) =
                copyto!(dest, LazyArrays.Mul(A,b))

            LinearAlgebra.mul!(dest::AbstractVector, A::$Mod{<:Any,<:$Typ}, b::AbstractVector) =
                copyto!(dest, LazyArrays.Mul(A,b))

            Base.:*(A::$Mod{<:Any,<:$Typ}, B::$Mod{<:Any,<:$Typ}, C...) = LazyArrays.apply(*,A,B, C...)
            Base.:*(A::$Mod{<:Any,<:$Typ}, B::AbstractMatrix, C...) = LazyArrays.apply(*,A,B, C...)
            Base.:*(A::AbstractMatrix, B::$Mod{<:Any,<:$Typ}, C...) = LazyArrays.apply(*,A,B, C...)
            Base.:*(A::$Mod{<:Any,<:$Typ}, B::AbstractVector, C...) = LazyArrays.apply(*,A,B, C...)
            Base.:*(A::$Mod{<:Any,<:$Typ}, B::LazyArrays.LazyVector, C...) = LazyArrays.apply(*,A,B, C...)

            Base.:*(A::$Mod{<:Any,<:$Typ}, B::$Typ, C...) = LazyArrays.apply(*,A,B, C...)
            Base.:*(A::$Typ, B::$Mod{<:Any,<:$Typ}, C...) = LazyArrays.apply(*,A,B, C...)

            Base.:*(A::$Mod{<:Any,<:$Typ}, B::Diagonal, C...) = LazyArrays.apply(*,A,B, C...)
            Base.:*(A::Diagonal, B::$Mod{<:Any,<:$Typ}, C...) = LazyArrays.apply(*,A,B, C...)

            Base.:*(A::LinearAlgebra.AbstractTriangular, B::$Mod{<:Any,<:$Typ}, C...) = LazyArrays.apply(*,A,B, C...)
            Base.:*(A::$Mod{<:Any,<:$Typ}, B::LinearAlgebra.AbstractTriangular, C...) = LazyArrays.apply(*,A,B, C...)
        end
        if Typ ≠ :LazyMatrix
            ret = quote
                $ret
                Base.:*(A::$Mod{<:Any,<:$Typ}, B::LazyArrays.LazyMatrix, C...) = LazyArrays.apply(*,A,B, C...)
            end
        end
    end

    esc(ret)
end

macro lazyldiv(Typ)
    esc(quote
        LinearAlgebra.ldiv!(A::$Typ, x::AbstractVector) = LazyArrays.materialize!(LazyArrays.Ldiv(A,x))
        LinearAlgebra.ldiv!(A::$Typ, x::AbstractMatrix) = LazyArrays.materialize!(LazyArrays.Ldiv(A,x))
        LinearAlgebra.ldiv!(A::$Typ, x::StridedVector) = LazyArrays.materialize!(LazyArrays.Ldiv(A,x))
        LinearAlgebra.ldiv!(A::$Typ, x::StridedMatrix) = LazyArrays.materialize!(LazyArrays.Ldiv(A,x))

        Base.:\(A::$Typ, x::AbstractVector) = LazyArrays.apply(\,A,x)
        Base.:\(A::$Typ, x::AbstractMatrix) = LazyArrays.apply(\,A,x)
    end)
end

@lazymul LazyMatrix
@lazylmul LazyMatrix
@lazyldiv LazyMatrix


*(A::AbstractMatrix, b::LazyVector) where T = apply(*,A,b)
*(A::Adjoint{<:Any,<:AbstractMatrix{T}}, b::LazyVector) where T = apply(*,A,b)