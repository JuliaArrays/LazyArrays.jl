colsupport(::TriangularLayout{'L'}, A, j) = colsupport(triangulardata(A), j) ∩ (minimum(j):size(A,1))
colsupport(::TriangularLayout{'U'}, A, j) = colsupport(triangulardata(A), j) ∩ OneTo(maximum(j))
rowsupport(::TriangularLayout{'U'}, A, j) = rowsupport(triangulardata(A), j) ∩ (minimum(j):size(A,2))
rowsupport(::TriangularLayout{'L'}, A, j) = rowsupport(triangulardata(A), j) ∩ OneTo(maximum(j))



###
# Lmul
###
mulapplystyle(::TriangularLayout, ::AbstractStridedLayout) = LmulStyle()





@inline function materialize!(M::BlasMatLmulVec{<:TriangularLayout{UPLO,UNIT,<:AbstractColumnMajor},
                                         <:AbstractStridedLayout}) where {UPLO,UNIT}
    A,x = M.A,M.B
    BLAS.trmv!(UPLO, 'N', UNIT, triangulardata(A), x)
end

@inline function materialize!(M::BlasMatLmulVec{<:TriangularLayout{'U',UNIT,<:AbstractRowMajor},
                                   <:AbstractStridedLayout}) where UNIT
    A,x = M.A,M.B
    BLAS.trmv!('L', 'T', UNIT, transpose(triangulardata(A)), x)
end

@inline function materialize!(M::BlasMatLmulVec{<:TriangularLayout{'L',UNIT,<:AbstractRowMajor},
                                   <:AbstractStridedLayout}) where UNIT
    A,x = M.A,M.B
    BLAS.trmv!('U', 'T', UNIT, transpose(triangulardata(A)), x)
end

@inline function materialize!(M::BlasMatLmulVec{<:TriangularLayout{'U',UNIT,<:ConjLayout{<:AbstractRowMajor}},
                                   <:AbstractStridedLayout,<:BlasComplex}) where UNIT
    A,x = M.A,M.B
    BLAS.trmv!('L', 'C', UNIT, triangulardata(A)', x)
end

@inline function materialize!(M::BlasMatLmulVec{<:TriangularLayout{'L',UNIT,<:ConjLayout{<:AbstractRowMajor}},
                                   <:AbstractStridedLayout,<:BlasComplex}) where UNIT
    A,x = M.A,M.B
    BLAS.trmv!('U', 'C', UNIT, triangulardata(A)', x)
end
# Triangular * Matrix

@inline function materialize!(M::BlasMatLmulMat{<:TriangularLayout{UPLO,UNIT,<:AbstractColumnMajor},
                                         <:AbstractStridedLayout, T}) where {UPLO,UNIT,T<:BlasFloat}
    A,x = M.A,M.B
    BLAS.trmm!('L', UPLO, 'N', UNIT, one(T), triangulardata(A), x)
end

@inline function materialize!(M::BlasMatLmulMat{<:TriangularLayout{'L',UNIT,<:AbstractRowMajor},
                                   <:AbstractStridedLayout, T}) where {UNIT,T<:BlasFloat}
    A,x = M.A,M.B
    BLAS.trmm!('L', 'U', 'T', UNIT, one(T), transpose(triangulardata(A)), x)
end

@inline function materialize!(M::BlasMatLmulMat{<:TriangularLayout{'U',UNIT,<:AbstractRowMajor},
    <:AbstractStridedLayout, T}) where {UNIT,T<:BlasFloat}
A,x = M.A,M.B
BLAS.trmm!('L', 'L', 'T', UNIT, one(T), transpose(triangulardata(A)), x)
end

@inline function materialize!(M::BlasMatLmulMat{<:TriangularLayout{'L',UNIT,<:ConjLayout{<:AbstractRowMajor}},
                                   <:AbstractStridedLayout, T}) where {UNIT,T<:BlasComplex}
    A,x = M.A,M.B
    BLAS.trmm!('L', 'U', 'C', UNIT, one(T), triangulardata(A)', x)
end

@inline function materialize!(M::BlasMatLmulMat{<:TriangularLayout{'U',UNIT,<:ConjLayout{<:AbstractRowMajor}},
                                                <:AbstractStridedLayout, T}) where {UNIT,T<:BlasComplex}
A,x = M.A,M.B
BLAS.trmm!('L', 'L', 'C', UNIT, one(T), triangulardata(A)', x)
end


materialize!(M::MatLmulMat{<:TriangularLayout}) = lmul!(M.A, M.B)

###
# Rmul
###

mulapplystyle(::AbstractStridedLayout, ::TriangularLayout) = RmulStyle()

@inline function materialize!(M::BlasMatRmulMat{<:AbstractStridedLayout,
                                                <:TriangularLayout{UPLO,UNIT,<:AbstractColumnMajor},T}) where {UPLO,UNIT,T<:BlasFloat}
    x,A = M.A,M.B
    BLAS.trmm!('R', UPLO, 'N', UNIT, one(T), triangulardata(A), x)
end

@inline function materialize!(M::BlasMatRmulMat{<:AbstractStridedLayout,
                                                <:TriangularLayout{'L',UNIT,<:AbstractRowMajor},T}) where {UNIT,T<:BlasFloat}
    x,A = M.A,M.B
    BLAS.trmm!('R', 'U', 'T', UNIT, one(T), transpose(triangulardata(A)), x)
end

@inline function materialize!(M::BlasMatRmulMat{<:AbstractStridedLayout,
                                                <:TriangularLayout{'U',UNIT,<:AbstractRowMajor},T}) where {UNIT,T<:BlasFloat}
x,A = M.A,M.B
BLAS.trmm!('R', 'L', 'T', UNIT, one(T), transpose(triangulardata(A)), x)
end

@inline function materialize!(M::BlasMatRmulMat{<:AbstractStridedLayout,
                                                <:TriangularLayout{'L',UNIT,<:ConjLayout{<:AbstractRowMajor}},T}) where {UPLO,UNIT,T<:BlasComplex}
    x,A = M.A,M.B
    BLAS.trmm!('R', 'U', 'C', UNIT, one(T), triangulardata(A)', x)
end

@inline function materialize!(M::BlasMatRmulMat{<:AbstractStridedLayout,
                                                <:TriangularLayout{'U',UNIT,<:ConjLayout{<:AbstractRowMajor}},T}) where {UPLO,UNIT,T<:BlasComplex}
x,A = M.A,M.B
BLAS.trmm!('R', 'L', 'C', UNIT, one(T), triangulardata(A)', x)
end


materialize!(M::MatRmulMat{<:AbstractStridedLayout,<:TriangularLayout}) = rmul!(M.A, M.B)


########
# Ldiv
########


@inline function copyto!(dest::AbstractArray, M::Ldiv{<:TriangularLayout})
    A, B = M.A, M.B
    dest ≡ B || (dest .= B)
    materialize!(Ldiv(A, dest))
end

for UNIT in ('U', 'N')
    for UPLO in ('L', 'U')
        @eval @inline materialize!(M::BlasMatLdivVec{<:TriangularLayout{$UPLO,$UNIT,<:AbstractColumnMajor},
                                            <:AbstractStridedLayout}) =
            BLAS.trsv!($UPLO, 'N', $UNIT, triangulardata(M.A), M.B)
    end

    @eval begin
        @inline materialize!(M::BlasMatLdivVec{<:TriangularLayout{'U',$UNIT,<:AbstractRowMajor},
                                                        <:AbstractStridedLayout}) =
            BLAS.trsv!('L', 'T', $UNIT, transpose(triangulardata(M.A)), M.B)

        @inline materialize!(M::BlasMatLdivVec{<:TriangularLayout{'L',$UNIT,<:AbstractRowMajor},
                                                        <:AbstractStridedLayout}) =
            BLAS.trsv!('U', 'T', $UNIT, transpose(triangulardata(M.A)), M.B)


        @inline materialize!(M::BlasMatLdivVec{<:TriangularLayout{'U',$UNIT,<:ConjLayout{<:AbstractRowMajor}},
                                                        <:AbstractStridedLayout}) =
            BLAS.trsv!('L', 'C', $UNIT, triangulardata(M.A)', M.B)

        @inline materialize!(M::BlasMatLdivVec{<:TriangularLayout{'L',$UNIT,<:ConjLayout{<:AbstractRowMajor}},
                                                        <:AbstractStridedLayout}) =
            BLAS.trsv!('U', 'C', $UNIT, triangulardata(M.A)', M.B)
    end
end

function materialize!(M::MatLdivMat{<:TriangularLayout})
    A,X = M.A,M.B
    size(A,2) == size(X,1) || thow(DimensionMismatch("Dimensions must match"))
    @views for j in axes(X,2)
        materialize!(Ldiv(A, X[:,j]))
    end
    X
end

function materialize!(M::MatLdivVec{<:TriangularLayout{'U','N'}})
    A,b = M.A,M.B
    require_one_based_indexing(A, b)
    n = size(A, 2)
    if !(n == length(b))
        throw(DimensionMismatch("second dimension of left hand side A, $n, and length of right hand side b, $(length(b)), must be equal"))
    end
    data = triangulardata(A)
    @inbounds for j in reverse(colsupport(b,1))
        iszero(data[j,j]) && throw(SingularException(j))
        bj = b[j] = data[j,j] \ b[j]
        for i in (1:j-1) ∩ colsupport(data,j)
            b[i] -= data[i,j] * bj
        end
    end
    b
end

function materialize!(M::MatLdivVec{<:TriangularLayout{'U','U'}})
    A,b = M.A,M.B
    require_one_based_indexing(A, b)
    n = size(A, 2)
    if !(n == length(b))
        throw(DimensionMismatch("second dimension of left hand side A, $n, and length of right hand side b, $(length(b)), must be equal"))
    end
    data = triangulardata(A)
    @inbounds for j in reverse(colsupport(b,1))
        iszero(data[j,j]) && throw(SingularException(j))
        bj = b[j]
        for i in (1:j-1) ∩ colsupport(data,j)
            b[i] -= data[i,j] * bj
        end
    end
    b
end

function materialize!(M::MatLdivVec{<:TriangularLayout{'L','N'}})
    A,b = M.A,M.B
    require_one_based_indexing(A, b)
    n = size(A, 2)
    if !(n == length(b))
        throw(DimensionMismatch("second dimension of left hand side A, $n, and length of right hand side b, $(length(b)), must be equal"))
    end
    data = triangulardata(A)
    @inbounds for j in 1:n
        iszero(data[j,j]) && throw(SingularException(j))
        bj = b[j] = data[j,j] \ b[j]
        for i in (j+1:n) ∩ colsupport(data,j)
            b[i] -= data[i,j] * bj
        end
    end
    b
end

function materialize!(M::BlasMatLdivVec{<:TriangularLayout{'L','U'}})
    A,b = M.A,M.B
    require_one_based_indexing(A, b)
    n = size(A, 2)
    if !(n == length(b))
        throw(DimensionMismatch("second dimension of left hand side A, $n, and length of right hand side b, $(length(b)), must be equal"))
    end
    data = triangulardata(A)
    @inbounds for j in 1:n
        iszero(data[j,j]) && throw(SingularException(j))
        bj = b[j]
        for i in (j+1:n) ∩ colsupport(data,j)
            b[i] -= data[i,j] * bj
        end
    end
    b
end
