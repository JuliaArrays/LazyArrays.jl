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
        Base.:*(A::AbstractMatrix, B::$Typ, C...) = LazyArrays.apply(*,A,B,C...)
        Base.:*(A::LinearAlgebra.AdjointAbsVec, B::$Typ, C...) = LazyArrays.apply(*,A,B,C...)
        Base.:*(A::LinearAlgebra.TransposeAbsVec, B::$Typ, C...) = LazyArrays.apply(*,A,B,C...)
    end
    if Typ ≠ :ApplyMatrix
        ret = quote
            $ret
            Base.:*(A::$Typ, B::LazyArrays.ApplyMatrix, C...) = LazyArrays.apply(*,A,B,C...)
            Base.:*(A::LazyArrays.ApplyMatrix, B::$Typ, C...) = LazyArrays.apply(*,A,B,C...)
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

            Base.:*(A::$Mod{<:Any,<:$Typ}, B::$Typ, C...) = LazyArrays.apply(*,A,B, C...)
            Base.:*(A::$Typ, B::$Mod{<:Any,<:$Typ}, C...) = LazyArrays.apply(*,A,B, C...)

            Base.:*(A::$Mod{<:Any,<:$Typ}, B::Diagonal, C...) = LazyArrays.apply(*,A,B, C...)
            Base.:*(A::Diagonal, B::$Mod{<:Any,<:$Typ}, C...) = LazyArrays.apply(*,A,B, C...)

            Base.:*(A::LinearAlgebra.AbstractTriangular, B::$Mod{<:Any,<:$Typ}, C...) = LazyArrays.apply(*,A,B, C...)
            Base.:*(A::$Mod{<:Any,<:$Typ}, B::LinearAlgebra.AbstractTriangular, C...) = LazyArrays.apply(*,A,B, C...)
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

macro lazyldiv(Typ)
    esc(quote
        LinearAlgebra.ldiv!(A::$Typ, x::AbstractVector) = (x .= LazyArrays.Ldiv(A,x))
        LinearAlgebra.ldiv!(A::$Typ, x::AbstractMatrix) = (x .= LazyArrays.Ldiv(A,x))
        LinearAlgebra.ldiv!(A::$Typ, x::StridedVector) = (x .= LazyArrays.Ldiv(A,x))
        LinearAlgebra.ldiv!(A::$Typ, x::StridedMatrix) = (x .= LazyArrays.Ldiv(A,x))

        Base.:\(A::$Typ, x::AbstractVector) = LazyArrays.apply(\,A,x)
        Base.:\(A::$Typ, x::AbstractMatrix) = LazyArrays.apply(\,A,x)
    end)
end

@lazymul ApplyMatrix
@lazylmul ApplyMatrix
@lazyldiv ApplyMatrix
@lazymul BroadcastMatrix
@lazylmul BroadcastMatrix
@lazyldiv BroadcastMatrix
