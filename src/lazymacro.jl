# Macros for lazy broadcasting,
# based on @dawbarton  https://discourse.julialang.org/t/19641/20
# and @tkf            https://github.com/JuliaLang/julia/issues/19198#issuecomment-457967851
# and @chethega      https://github.com/JuliaLang/julia/pull/30939

using MacroTools

lazy(::Any) = throw(ArgumentError("function `lazy` exists only for its effect on broadcasting, see the macro @~"))
struct LazyCast{T}
    value::T
end
Broadcast.broadcasted(::typeof(lazy), x) = LazyCast(x)
Broadcast.materialize(x::LazyCast) = BroadcastArray(x.value)

"""
    @~ expr

Macro for creating lazy `BroadcastArray`s.
Expects a broadcasting expression, possibly created by the `@.` macro:
```
julia> @~ A .+ B ./ 2

julia> @~ @. A + B / 2
```
"""
macro ~(ex)
    checkex(ex)
    esc( :( $lazy.($ex) ) )
end

function checkex(ex)
    if @capture(ex, (arg__,) = val_ )
        if arg[2]==:dims
            throw(ArgumentError("@~ is capturing keyword arguments, try with `; dims = $val` instead of a comma"))
        else
            throw(ArgumentError("@~ is probably capturing capturing keyword arguments, try with ; or brackets"))
        end
    end
    if @capture(ex, (arg_,rest__) )
        throw(ArgumentError("@~ is capturing more than one expression, try $name($arg) with brackets"))
    end
    ex
end
