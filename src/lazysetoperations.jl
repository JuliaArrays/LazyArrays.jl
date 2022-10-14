

eltype(A::Applied{<:Any,typeof(union)}) = mapreduce(eltype,promote_type,A.args)
eltype(A::Applied{<:Any,typeof(intersect)}) = mapreduce(eltype,promote_type,A.args)
eltype(A::Applied{<:Any,typeof(setdiff)}) = eltype(first(A.args))

struct VectorSetStyle <: ApplyStyle end
struct SetStyle <: ApplyStyle end




for func in (:union, :intersect, :setdiff)
    @eval ApplyStyle(::typeof($func), A...) = combine_set_styles(A...)
end



# Code adapted from julia/base/broadcast.jl

# combine_styles operates on values (arbitrarily many)
combine_set_styles() = VectorSetStyle()
combine_set_styles(::Type{<:AbstractSet}) = SetStyle()
combine_set_styles(::Type) = VectorSetStyle()
combine_set_styles(c1, c2) = result_set_style(combine_set_styles(c1), combine_set_styles(c2))
@inline combine_set_styles(c1, c2, cs...) = result_set_style(combine_set_styles(c1), combine_set_styles(c2, cs...))

# result_set_style works on types (singletons and pairs), and leverages `Style`
result_set_style(s::ApplyStyle) = s
result_set_style(s1::S, s2::S) where S<:ApplyStyle = S()
# Test both orders so users typically only have to declare one order
result_set_style(::VectorSetStyle, ::SetStyle) = SetStyle()
result_set_style(::SetStyle, ::VectorSetStyle) = SetStyle()


emptymutable(::Applied{SetStyle}, ::Type{T}) where T = Set{T}()
emptymutable(::Applied{VectorSetStyle}, ::Type{T}) where T = Vector{T}()
emptymutable(A::Applied) = emptymutable(A, eltype(A))



copy(u::Applied{<:Any,typeof(union)}) = union!(emptymutable(u), u.args...)
copy(u::Applied{<:Any,typeof(intersect)}) = intersect!(emptymutable(u), u.args...)
copy(u::Applied{<:Any,typeof(setdiff)}) = setdiff!(emptymutable(u), u.args...)

function in(x, u::Applied{<:Any,typeof(union)})
    for d in u.args
        x ∈ d && return true
    end
    return false
end

function in(x, u::Applied{<:Any,typeof(intersect)})
    for d in u.args
        x ∈ d || return false
    end
    return true
end

function in(x, u::Applied{<:Any,typeof(setdiff)})
    x ∈ first(u.args) || return false
    for d in tail(u.args)
        x ∈ d && return false
    end
    return true
end



for func in (:union, :intersect, :setdiff)
    @eval begin
        function issubset(l, r::Applied{<:Any,typeof($func)})
            for x in l
                x ∈ r || return false
            end
            return true
        end
    end
end
