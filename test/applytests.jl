using LazyArrays, Test
import LazyArrays: materialize, broadcasted, DefaultApplyStyle, Applied

@test applied(exp,1) isa Applied{DefaultApplyStyle}

materialize(applied(randn))

@test materialize(applied(*, 1)) == 1

@test materialize(applied(exp, 1)) === exp(1)
@test materialize(applied(exp, broadcasted(+, 1, 2))) ===
      materialize(applied(exp, applied(+, 1, 2))) === exp(3)
