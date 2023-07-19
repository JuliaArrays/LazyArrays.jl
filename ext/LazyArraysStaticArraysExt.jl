module LazyArraysStaticArraysExt

using LazyArrays
using LazyArrays: LazyArrayStyle
using StaticArrays
using StaticArrays: StaticArrayStyle

function LazyArrays._vcat_layout_broadcasted((Ahead,Atail)::Tuple{SVector{M},Any},
				(Bhead,Btail)::Tuple{SVector{M},Any}, op, A, B) where M
	Vcat(op.(Ahead,Bhead), op.(Atail,Btail))
end

Base.BroadcastStyle(L::LazyArrayStyle{N}, ::StaticArrayStyle{N}) where N = L

end
