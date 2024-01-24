module AStat

using StatsBase
using DataStructures
using LinearAlgebra

export ols, lis, median_filter, randf, Spearman, freqmap

function ols( X, y )
    n = length(y)
    inv = pinv(X'*X)
    beta = inv*X'*y
    yhat = X*beta
    res = y - yhat
    var = res'*res/n
    yvar = y'*y/n
    return Dict(
        :beta => beta,
        :var => var,
        :yvar => yvar,
        :std => sqrt(var),
        :covbeta => var * inv,
        :R2 => 1 - var/yvar,
        :yhat => yhat,
        :res => res,
    )
end

function lis( x::AbstractVector{T} ) where {T}
    n = length(x)
    indices = fill( 0, n )
    parent = fill( 0, n )
    l = 0
    for i = 1:n
        r = binary_search( indices, x, x[i], l )
        if r > l
            l = r
            indices[r] = i
        elseif x[i] < x[indices[r]]
            indices[r] = i
        end
        if r > 1
            parent[i] = indices[r-1]
        end
    end

    lis = fill( 0, l )
    i = l
    lis[i] = indices[l]
    while parent[lis[i]] != 0
        lis[i-1] = parent[lis[i]]
        i -= 1
    end
    return lis
end

function binary_search( indices, a, x, n )
    if n == 0
        return 1
    end
    lo = 1
    hi = n+1
    while lo < hi
        mid = div( lo + hi, 2 )
        if x <= a[indices[mid]]
            hi = mid
        else
            lo = mid+1
        end
    end
    return lo
end

function median_filter( x::Vector{T}, n::Int ) where {T}
    N = length(x)
    result = Array{T}( undef, N - n + 1 );
    for i = 1:N-n+1
        result[i] = median( x[i:i+n-1] )
    end
    return result
end

function randf( f, x, y, n )
    maxf = -Inf
    for i = 1:n
        sy = shuffle(y)
        c = f( x, sy )
        if c > maxf
            maxf = c
        end
    end
    return maxf
end

Spearman(x, y) = cor(invperm(sortperm(x)), invperm(sortperm(y)))

function freqmap( a::AbstractVector{T} ) where {T}
    d = Dict{T,Int}()
    for x in a
        d[x] = get( d, x, 0 ) + 1
    end
    ks = collect(keys(d))
    vs = collect(values(d))
    sp = sortperm(vs, order=Base.Order.Reverse)
    return OrderedDict( zip( ks[sp], vs[sp]./sum(vs) ) )
end

end # module AStat
