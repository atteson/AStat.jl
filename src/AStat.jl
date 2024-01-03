module AStat

export ols, lis

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

end # module AStat
