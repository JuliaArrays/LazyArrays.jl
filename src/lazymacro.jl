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
Broadcast.materialize(x::LazyCast) = x.value


is_call(ex) = isexpr(ex, :call) && !is_dotcall(ex)

is_dotcall(ex) =
    (isexpr(ex, :.) && isexpr(ex.args[2], :tuple)) ||
    (isexpr(ex, :call) && ex.args[1] isa Symbol && startswith(String(ex.args[1]), "."))
# e.g., `f.(x, y, z)` or `x .+ y .+ z`

lazy_expr(x) = x
function lazy_expr(ex::Expr)
    if is_dotcall(ex)
        return bc_expr(ex)
    elseif is_call(ex)
        return app_expr(ex)
    else
        # TODO: Maybe better to support `a ? b : c` etc.? But how?
        return ex
    end
end

function bc_expr(ex::Expr)
    @assert is_dotcall(ex)
    return :($lazy.($(bc_expr_impl(ex))))
end

bc_expr_impl(x) = x
function bc_expr_impl(ex::Expr)
    # walk down chain of dot calls
    if ex.head == :. && ex.args[2].head === :tuple
        @assert length(ex.args) == 2  # argument is always expressed as a tuple
        f = ex.args[1]  # function name
        args = ex.args[2].args
        return Expr(ex.head, lazy_expr(f), Expr(:tuple, bc_expr_impl.(args)...))
    elseif ex.head == :call && startswith(String(ex.args[1]), ".")
        f = ex.args[1]  # function name (e.g., `.+`)
        args = ex.args[2:end]
        return Expr(ex.head, lazy_expr(f), bc_expr_impl.(args)...)
    else
        @assert !is_dotcall(ex)
        return lazy_expr(ex)
    end
end

function app_expr(ex::Expr)
    @assert is_call(ex)
    return app_expr_impl(ex)
end

app_expr_impl(x) = x
function app_expr_impl(ex::Expr)
    # walk down chain of calls and lazy-ify them
    if is_call(ex)
        if isexpr(ex.args[1], :$)
            # eagerly evaluate the call
            return Expr(:call, ex.args[1].args[1], ex.args[2:end]...)
        end
        return :($applied($(app_expr_impl.(ex.args)...)))
    else
        return lazy_expr(ex)
    end
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

"""
    @~ expr

Macro for creating a `Broadcasted` or `Applied` object.  Regular calls
like `f(args...)` inside `expr` are replaced with `applied(f, args...)`.
Dotted-calls like `f(args...)` inside `expr` are replaced with
`broadcasted.(f, args...)`.  Use `LazyArray(@~ expr)` if you need an
array-based interface.

```
julia> @~ A .+ B ./ 2

julia> @~ @. A + B / 2

julia> @~ A * B + C
```
"""
macro ~(ex)
    checkex(ex)
    # Expanding macro here to support, e.g., `@.`
    esc(:($instantiate($(lazy_expr(macroexpand(__module__, ex))))))
end
