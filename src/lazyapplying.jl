



abstract type ApplyStyle end
struct DefaultApplyStyle <: ApplyStyle end
struct LayoutApplyStyle{Layouts<:Tuple} <: ApplyStyle
    layouts::Layouts
end

ApplyStyle(f, args...) = DefaultApplyStyle()

struct Applied{Style<:ApplyStyle, F, Args<:Tuple}
    style::Style
    f::F
    args::Args
end

applied(f, args...) = Applied(ApplyStyle(f, args...), f, args)
materialize(A::Applied{DefaultApplyStyle}) = A.f(materialize.(A.args)...)
