module SetOperations

import Base: eltype, tail, in, issubset

abstract type Style end
struct VectorStyle <: Style end
struct SetStyle <: Style end
struct Unknown <: Style end

Style(::Type{<:AbstractSet}) = SetStyle()
Style(::Type) = VectorStyle()

Style(::S, ::S) where S<:Style = S() # homogeneous types preserved
# Fall back to Unknown. This is necessary to implement argument-swapping
Style(::Style, ::Style) = Unknown()
# Unknown loses to everything
Style(::Unknown, ::Unknown) = Unknown()
Style(::SetStyle, ::VectorStyle) = SetStyle()


# Code adapted from julia/base/broadcast.jl

# combine_styles operates on values (arbitrarily many)
combine_styles() = VectorStyle()
combine_styles(c) = result_style(Style(typeof(c)))
combine_styles(c1, c2) = result_style(combine_styles(c1), combine_styles(c2))
@inline combine_styles(c1, c2, cs...) = result_style(combine_styles(c1), combine_styles(c2, cs...))

# result_style works on types (singletons and pairs), and leverages `Style`
result_style(s::Style) = s
result_style(s1::S, s2::S) where S<:Style = S()
# Test both orders so users typically only have to declare one order
result_style(s1, s2) = result_join(s1, s2, Style(s1, s2), Style(s2, s1))

# result_join is the final arbiter. Because `Style` for undeclared pairs results in Unknown,
# we defer to any case where the result of `Style` is known.
result_join(::Any, ::Any, ::Unknown, ::Unknown)   = Unknown()
result_join(::Any, ::Any, ::Unknown, s::Style) = s
result_join(::Any, ::Any, s::Style, ::Unknown) = s
# For AbstractArray types with specialized broadcasting and undefined precedence rules,
# we have to signal conflict. Because ArrayConflict is a subtype of AbstractArray,
# this will "poison" any future operations (if we instead returned `DefaultArrayStyle`, then for
# 3-array broadcasting the returned type would depend on argument order).
result_join(::VectorStyle, ::VectorStyle, ::Unknown, ::Unknown) =
    ArrayConflict()
# Fallbacks in case users define `rule` for both argument-orders (not recommended)
result_join(::Any, ::Any, ::S, ::S) where S<:Style = S()
@noinline function result_join(::S, ::T, ::U, ::V) where {S,T,U,V}
    error("""
conflicting broadcast rules defined
  LazyArays.Style(::$S, ::$T) = $U()
  LazyArays.Style(::$T, ::$S) = $V()
One of these should be undefined (and thus return Unknown).""")
end

abstract type SetOperation{Style} end

for Typ in (:Unioned, :Intersected, :SetDiffed)
    @eval begin
        struct $Typ{Style, Args<:Tuple} <: SetOperation{Style}
            args::Args
        end

        $Typ(args) = $Typ{typeof(combine_styles(args...))}(args)
        $Typ{Style}(args) where Style = $Typ{Style, typeof(args)}(args)
    end
end

eltype(u::SetOperation) = mapreduce(eltype, promote_type, u.args)

emptymutable(::SetOperation{SetStyle}, ::Type{T}) where T = Set{T}()
emptymutable(::SetOperation{VectorStyle}, ::Type{T}) where T = Vector{T}()

emptymutable(u::SetOperation) = emptymutable(u, eltype(u))

materialize(u::SetOperation) = copy(u)

unioned(args...) = Unioned(args)
union(a...) = materialize(unioned(a...))
intersected(args...) = Intersected(args)
intersect(a...) = materialize(intersected(a...))
setdiffed(args...) = SetDiffed(args)
setdiff(a...) = materialize(setdiffed(a...))

const ∪ = union
const ∩ = intersect

copy(u::Unioned) = union!(emptymutable(u), u.args...)
copy(u::Intersected) = intersect!(emptymutable(u), u.args...)
copy(u::SetDiffed) = setdiff!(emptymutable(u), u.args...)

function in(x, u::Unioned)
    for d in u.args
        x ∈ d && return true
    end
    return false
end

function in(x, u::Intersected)
    for d in u.args
        x ∈ d || return false
    end
    return true
end

function in(x, u::SetDiffed)
    x ∈ first(u.args) || return false
    for d in tail(u.args)
        x ∈ d && return false
    end
    return true
end

function issubset(l, r::SetOperation)
    for x in l
        x ∈ r || return false
    end
    return true
end

end # SetOperations
