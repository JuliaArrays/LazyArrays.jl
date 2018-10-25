struct ArrayMulArrayStyle{StyleA, StyleB, p, q} <: BroadcastStyle end

@inline copyto!(dest::AbstractArray, bc::Broadcasted{<:ArrayMulArrayStyle}) =
    _copyto!(MemoryLayout(dest), dest, bc)
# Use default broacasting in general
@inline _copyto!(_, dest, bc::Broadcasted) = copyto!(dest, Broadcasted{Nothing}(bc.f, bc.args, bc.axes))

const BArrayMulArray{styleA, styleB, p, q, T, V} =
    Broadcasted{ArrayMulArrayStyle{styleA,styleB,p,q}, <:Any, typeof(identity),
                <:Tuple{<:ArrayMulArray{styleA,styleB,p,q,T,V}}}
const BConstArrayMulArray{T, styleA, styleB, p, q} =
    Broadcasted{ArrayMulArrayStyle{styleA,styleB,p,q},
                    <:Any, typeof(*),
                    <:Tuple{T,<:ArrayMulArray{styleA,styleB,p,q,T,T}}}
const BArrayMulArrayPlusArray{T, styleA, styleB, p, q} =
    Broadcasted{ArrayMulArrayStyle{styleA,styleB,p,q},
                <:Any, typeof(+),
                <:Tuple{<:ArrayMulArray{styleA,styleB,p,q,T,T},<:AbstractArray{T,q}}}
const BArrayMulArrayPlusConstArray{T, styleA, styleB, p, q} =
    Broadcasted{ArrayMulArrayStyle{styleA,styleB,p,q},
                <:Any, typeof(+),
                <:Tuple{<:ArrayMulArray{styleA,styleB,p,q,T,T},
                Broadcasted{DefaultArrayStyle{q},<:Any,typeof(*),
                            <:Tuple{T,<:AbstractArray{T,q}}}}}
const BConstArrayMulArrayPlusArray{T, styleA, styleB, p, q} =
    Broadcasted{ArrayMulArrayStyle{styleA,styleB,p,q},
                <:Any, typeof(+),
                <:Tuple{Broadcasted{<:ArrayMulArrayStyle{styleA,styleB,p,q},
                                    <:Any, typeof(*),
                                    <:Tuple{T,<:ArrayMulArray{styleA,styleB,p,q,T,T}}},
                        <:AbstractArray{T,q}}}
const BConstArrayMulArrayPlusConstArray{T, styleA, styleB, p, q} =
    Broadcasted{ArrayMulArrayStyle{styleA,styleB,p,q},
                <:Any, typeof(+),
                <:Tuple{Broadcasted{<:ArrayMulArrayStyle{styleA,styleB,p,q},
                                    <:Any, typeof(*),
                                    <:Tuple{T,<:ArrayMulArray{styleA,styleB,p,q,T,T}}},
                        Broadcasted{DefaultArrayStyle{q},<:Any,typeof(*),<:Tuple{T,<:AbstractArray{T,q}}}}}


BroadcastStyle(::Type{<:ArrayMulArray{StyleA,StyleB,p,q}}) where {StyleA,StyleB,p,q} =
    ArrayMulArrayStyle{StyleA,StyleB,p,q}()
BroadcastStyle(M::ArrayMulArrayStyle, ::DefaultArrayStyle) = M
BroadcastStyle(::DefaultArrayStyle, M::ArrayMulArrayStyle) = M
similar(M::Broadcasted{<:ArrayMulArrayStyle}, ::Type{ElType}) where ElType =
    Array{Eltype}(undef,size(M.args[1]))

# Use copyto! for y .= Mul(A,b)
@inline function _copyto!(_, dest::AbstractArray, bc::BArrayMulArray)
    (M,) = bc.args
    copyto!(dest, M)
end


####
# Matrix * Vector
####
let (p,q) = (2,1)
    global const MatMulVecStyle{StyleA, StyleB} = ArrayMulArrayStyle{StyleA, StyleB, p, q}

    global const BConstMatVec{T, styleA, styleB} = BConstArrayMulArray{T, styleA, styleB, p, q}
    global const BMatVecPlusVec{T,styleA,styleB} = BArrayMulArrayPlusArray{T, styleA, styleB, p, q}
    global const BMatVecPlusConstVec{T,styleA,styleB} = BArrayMulArrayPlusConstArray{T, styleA, styleB, p, q}
    global const BConstMatVecPlusVec{T, styleA, styleB} = BConstArrayMulArrayPlusArray{T, styleA, styleB, p, q}
    global const BConstMatVecPlusConstVec{T, styleA, styleB} = BConstArrayMulArrayPlusConstArray{T, styleA, styleB, p, q}
end

broadcastable(M::MatMulVec) = M
instantiate(bc::Broadcasted{<:MatMulVecStyle}) = bc


# @inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
#          bc::BMatVec{T, <:AbstractColumnMajor, <:AbstractStridedLayout}) where T<: BlasFloat
#     (M,) = bc.args
#     dest .= one(T) .* M .+ zero(T) .* dest
# end


####
# Matrix * Matrix
####


let (p,q) = (2,2)
    global const MatMulMatStyle{StyleA, StyleB} = ArrayMulArrayStyle{StyleA, StyleB, p, q}

    global const BMatMat{styleA, styleB, T, V} = BArrayMulArray{styleA, styleB, p, q, T, V}
    global const BConstMatMat{T, styleA, styleB} = BConstArrayMulArray{T, styleA, styleB, p, q}
    global const BMatMatPlusMat{T,styleA,styleB} = BArrayMulArrayPlusArray{T, styleA, styleB, p, q}
    global const BMatMatPlusConstMat{T,styleA,styleB} = BArrayMulArrayPlusConstArray{T, styleA, styleB, p, q}
    global const BConstMatMatPlusMat{T, styleA, styleB} = BConstArrayMulArrayPlusArray{T, styleA, styleB, p, q}
    global const BConstMatMatPlusConstMat{T, styleA, styleB} = BConstArrayMulArrayPlusConstArray{T, styleA, styleB, p, q}
end

broadcastable(M::MatMulMat) = M
instantiate(bc::Broadcasted{<:MatMulMatStyle}) = bc



macro _blasmatvec(Lay, Typ)
    esc(quote
        # y .= Mul(A,b) gets lowered here
        @inline function LazyArrays._copyto!(::LazyArrays.AbstractStridedLayout, dest::AbstractVector{T},
                                             M::LazyArrays.MatMulVec{$Lay, <:LazyArrays.AbstractStridedLayout, T, T}) where T<: $Typ
            A,B = M.A, M.B
            materialize!(BLASMul(one(T), A, B, zero(T), dest))
        end

        @inline function LazyArrays._copyto!(::LazyArrays.AbstractStridedLayout, dest::AbstractVector{T},
                 bc::LazyArrays.BConstMatVec{T, $Lay,<:LazyArrays.AbstractStridedLayout}) where T<: $Typ
            α,M = bc.args
            A,B = M.A, M.B
            materialize!(BLASMul(α, A, B, zero(T), dest))
        end

        @inline function LazyArrays._copyto!(::LazyArrays.AbstractStridedLayout, dest::AbstractVector{T},
                 bc::LazyArrays.BMatVecPlusVec{T,$Lay,<:LazyArrays.AbstractStridedLayout}) where T<: $Typ
            M,C = bc.args
            A,B = M.A, M.B
            copyto!(dest, BLASMul(one(T), A, B, one(T), C))
        end

        @inline function LazyArrays._copyto!(::LazyArrays.AbstractStridedLayout, dest::AbstractVector{T},
                 bc::LazyArrays.BMatVecPlusConstVec{T,$Lay,<:LazyArrays.AbstractStridedLayout}) where T<: $Typ
            M,βc = bc.args
            β,C = βc.args
            A,B = M.A, M.B
            copyto!(dest, BLASMul(one(T), A, B, β, C))
        end

        @inline function LazyArrays._copyto!(::LazyArrays.AbstractStridedLayout, dest::AbstractVector{T},
                 bc::LazyArrays.BConstMatVecPlusVec{T,$Lay,<:LazyArrays.AbstractStridedLayout}) where T<: $Typ
            αM,C = bc.args
            α,M = αM.args
            A,B = M.A, M.B
            copyto!(dest, BLASMul(α, A, B, one(T), C))
        end

        @inline function LazyArrays._copyto!(::LazyArrays.AbstractStridedLayout, dest::AbstractVector{T},
                 bc::LazyArrays.BConstMatVecPlusConstVec{T,$Lay,<:LazyArrays.AbstractStridedLayout}) where T<: $Typ
            αM,βc = bc.args
            α,M = αM.args
            A,B = M.A, M.B
            β,C = βc.args
            copyto!(dest, BLASMul(α, A, B, β, C))
        end
    end)
end

macro blasmatvec(Lay)
    esc(quote
        LazyArrays.@_blasmatvec $Lay LinearAlgebra.BLAS.BlasFloat
        LazyArrays.@_blasmatvec LazyArrays.ConjLayout{$Lay} LinearAlgebra.BLAS.BlasComplex
    end)
end


# Matrix * Vector

# this should work but for some reason was not inlining correctly
# @inline function _copyto!(::AbstractStridedLayout, dest::AbstractVector,
#          bc::BMatVec{T, <:AbstractColumnMajor, <:AbstractStridedLayout}) where T<: BlasFloat
#     (M,) = bc.args
#     dest .= one(T) .* M .+ zero(T) .* dest
# end

macro _blasmatmat(CTyp, ATyp, BTyp, Typ)
    esc(quote
        @inline function LazyArrays._copyto!(::$CTyp, dest::AbstractMatrix{T},
                 M::LazyArrays.MatMulMat{$ATyp,$BTyp,T,T}) where T<: $Typ
            A,B = M.A, M.B
            materialize!(BLASMul(one(T), A, B, zero(T), dest))
        end

        @inline function LazyArrays._copyto!(::$CTyp, dest::AbstractMatrix{T},
                 bc::LazyArrays.BConstMatMat{T,$ATyp,$BTyp}) where T<: $Typ
            α,M = bc.args
            A,B = M.A, M.B
            materialize!(BLASMul(α, A, B, zero(T), dest))
        end

        @inline function LazyArrays._copyto!(::$CTyp, dest::AbstractMatrix{T},
                 bc::LazyArrays.BMatMatPlusMat{T,$ATyp,$BTyp}) where T<: $Typ
            M,C = bc.args
            A,B = M.A, M.B
            copyto!(dest, BLASMul(one(T), A, B, one(T), C))
        end

        @inline function LazyArrays._copyto!(::$CTyp, dest::AbstractMatrix{T},
                 bc::LazyArrays.BMatMatPlusConstMat{T,$ATyp,$BTyp}) where T<: $Typ
            M,βc = bc.args
            β,C = βc.args
            A,B = M.A, M.B
            copyto!(dest, BLASMul(one(T), A, B, β, C))
        end

        @inline function LazyArrays._copyto!(::$CTyp, dest::AbstractMatrix{T},
                 bc::LazyArrays.BConstMatMatPlusMat{T,$ATyp,$BTyp}) where T<: $Typ
            αM,C = bc.args
            α,M = αM.args
            A,B = M.A, M.B
            copyto!(dest, BLASMul(α, A, B, one(T), C))
        end

        @inline function LazyArrays._copyto!(::$CTyp, dest::AbstractMatrix{T},
                 bc::LazyArrays.BConstMatMatPlusConstMat{T,$ATyp,$BTyp}) where T<: $Typ
            αM,βc = bc.args
            α,M = αM.args
            A,B = M.A, M.B
            β,C = βc.args
            copyto!(dest, BLASMul(α, A, B, β, C))
        end
    end)
end

macro blasmatmat(ATyp, BTyp, CTyp)
    esc(quote
        LazyArrays.@_blasmatmat $ATyp $BTyp $CTyp BlasFloat
        LazyArrays.@_blasmatmat LazyArrays.ConjLayout{$ATyp} $BTyp $CTyp BlasComplex
        LazyArrays.@_blasmatmat $ATyp LazyArrays.ConjLayout{$BTyp} $CTyp BlasComplex
        LazyArrays.@_blasmatmat LazyArrays.ConjLayout{$ATyp} LazyArrays.ConjLayout{$BTyp} $CTyp BlasComplex
        LazyArrays.@_blasmatmat $ATyp $BTyp LazyArrays.ConjLayout{$CTyp} BlasComplex
        LazyArrays.@_blasmatmat LazyArrays.ConjLayout{$ATyp} $BTyp LazyArrays.ConjLayout{$CTyp} BlasComplex
        LazyArrays.@_blasmatmat $ATyp LazyArrays.ConjLayout{$BTyp} LazyArrays.ConjLayout{$CTyp} BlasComplex
        LazyArrays.@_blasmatmat LazyArrays.ConjLayout{$ATyp} LazyArrays.ConjLayout{$BTyp} LazyArrays.ConjLayout{$CTyp} BlasComplex
    end)
end


@blasmatvec ColumnMajor
@blasmatvec DenseColumnMajor
@blasmatvec RowMajor
@blasmatvec DenseRowMajor


@blasmatmat AbstractColumnMajor DenseColumnMajor DenseColumnMajor
@blasmatmat AbstractColumnMajor DenseColumnMajor ColumnMajor
@blasmatmat AbstractColumnMajor DenseColumnMajor DenseRowMajor
@blasmatmat AbstractColumnMajor DenseColumnMajor RowMajor
@blasmatmat AbstractColumnMajor ColumnMajor DenseColumnMajor
@blasmatmat AbstractColumnMajor ColumnMajor ColumnMajor
@blasmatmat AbstractColumnMajor ColumnMajor DenseRowMajor
@blasmatmat AbstractColumnMajor ColumnMajor RowMajor
@blasmatmat AbstractColumnMajor DenseRowMajor DenseColumnMajor
@blasmatmat AbstractColumnMajor DenseRowMajor ColumnMajor
@blasmatmat AbstractColumnMajor DenseRowMajor DenseRowMajor
@blasmatmat AbstractColumnMajor DenseRowMajor RowMajor
@blasmatmat AbstractColumnMajor RowMajor DenseColumnMajor
@blasmatmat AbstractColumnMajor RowMajor ColumnMajor
@blasmatmat AbstractColumnMajor RowMajor DenseRowMajor
@blasmatmat AbstractColumnMajor RowMajor RowMajor

@blasmatmat AbstractRowMajor DenseColumnMajor DenseColumnMajor
@blasmatmat AbstractRowMajor DenseColumnMajor ColumnMajor
@blasmatmat AbstractRowMajor DenseColumnMajor DenseRowMajor
@blasmatmat AbstractRowMajor DenseColumnMajor RowMajor
@blasmatmat AbstractRowMajor ColumnMajor DenseColumnMajor
@blasmatmat AbstractRowMajor ColumnMajor ColumnMajor
@blasmatmat AbstractRowMajor ColumnMajor DenseRowMajor
@blasmatmat AbstractRowMajor ColumnMajor RowMajor
@blasmatmat AbstractRowMajor DenseRowMajor DenseColumnMajor
@blasmatmat AbstractRowMajor DenseRowMajor ColumnMajor
@blasmatmat AbstractRowMajor DenseRowMajor DenseRowMajor
@blasmatmat AbstractRowMajor DenseRowMajor RowMajor
@blasmatmat AbstractRowMajor RowMajor DenseColumnMajor
@blasmatmat AbstractRowMajor RowMajor ColumnMajor
@blasmatmat AbstractRowMajor RowMajor DenseRowMajor
@blasmatmat AbstractRowMajor RowMajor RowMajor
