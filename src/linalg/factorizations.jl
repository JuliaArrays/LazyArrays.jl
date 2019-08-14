

ApplyStyle(::typeof(*), ::Type{<:AbstractQ}, ::Type{<:AbstractMatrix}) = DefaultApplyStyle()
ApplyStyle(::typeof(*), ::Type{<:AbstractQ}, ::Type{<:AbstractVector}) = DefaultApplyStyle()

# we need to special case AbstractQ as it allows non-compatiple multiplication
function check_mul_axes(A::AbstractQ, B, C...) 
    axes(A.factors, 1) == axes(B, 1) || axes(A.factors, 2) == axes(B, 1) ||  
        throw(DimensionMismatch("First axis of B, $(axes(B,1)) must match either axes of A, $(axes(A))"))
    check_mul_axes(B, C...)
end