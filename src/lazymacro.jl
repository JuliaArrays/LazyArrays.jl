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

""" Check if `ex` is a dot-call expression like `f.(x, y, z)` or `x .+ y .+ z`. """
is_dotcall(ex) = is_dotcall_nonop(ex) || is_dotcall_op(ex)

""" Check if `ex` is an expression like `f.(x, y, z)`. """
is_dotcall_nonop(ex) =
    isexpr(ex, :.) && length(ex.args) == 2 && isexpr(ex.args[2], :tuple)

""" Check if `ex` is an expression like `x .+ y .+ z`. """
function is_dotcall_op(ex)
    ex isa Expr && !isempty(ex.args) || return false
    op = ex.args[1]
    return op isa Symbol && Base.isoperator(op) && startswith(string(op), ".")
end

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
    if is_dotcall_nonop(ex)
        f = ex.args[1]  # function name
        args = ex.args[2].args
        return Expr(ex.head, lazy_expr(f), Expr(:tuple, bc_expr_impl.(args)...))
    elseif is_dotcall_op(ex)
        f = ex.args[1]  # function name (e.g., `.+`)
        args = ex.args[2:end]
        return Expr(ex.head, lazy_expr(f), bc_expr_impl.(args)...)
    else
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
        throw(ArgumentError("@~ is capturing more than one expression, try capturing \"$arg\" with brackets"))
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
