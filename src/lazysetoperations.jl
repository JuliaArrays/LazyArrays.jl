abstract type UnionStyle end
struct DefaultUnionStyle <: UnionStyle end

UnionStyle(_) = DefaultUnionStyle()


struct Unioned{StyleA, DD}
    domains::DD
end

Unioned(domains...) = Unioned{mapreduce(eltype,promote_type,domains),typeof(domains)}(domains)

function in(x, u::Unioned)
    for d in u.domains
        x ∈ d && return true
    end
    return false
end

∪(a, b) = Unioned(a, b)



#
# struct Intersected{T,DD}
#     domains::D
# end
#
# Intersected(domains...) = Intersected{mapreduce(eltype,promote_type,domains),typeof(domains)}(domains)
#
# function in(x, u::Unioned)
#     for d in u.domains
#         x ∈ d && return true
#     end
#     return false
# end
