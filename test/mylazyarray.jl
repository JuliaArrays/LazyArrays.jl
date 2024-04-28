struct MyLazyArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end


Base.size(A::MyLazyArray) = size(A.data)
Base.getindex(A::MyLazyArray, j::Int...) = A.data[j...]
LazyArrays.MemoryLayout(::Type{<:MyLazyArray}) = LazyLayout()
Base.BroadcastStyle(::Type{<:MyLazyArray{<:Any,N}}) where N = LazyArrayStyle{N}()
LinearAlgebra.factorize(A::MyLazyArray) = factorize(A.data)
