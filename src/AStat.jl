module AStat

include("CG.jl")

using StatsBase
using DataStructures
using LinearAlgebra
using Distributions

import StatsBase: mean, std, var

export OLSModel, FTest, pvalue
export lis, median_filter, randf, Spearman, freqmap, momentmap, label, bound, multimomentmap

struct OLSModel{T <: Number, M <: AbstractMatrix{T},V <: AbstractVector{T}}
    X::M
    y::V
    beta::Vector{T}
    rss::T
    std::T
    covbeta::Matrix{T}
    R2::T
    yhat::Vector{T}
    res::Vector{T}
end

function OLSModel( X::M, y::V ) where {T <: Number, M <: AbstractMatrix{T}, V <: AbstractVector{T}}
    n = length(y)
    inv = pinv(X'*X)
    beta = inv*X'*y
    yhat = X*beta
    res = y - yhat
    rss_model = res'*res
    rss = y'*y
    var = rss_model/n
    return OLSModel(
        X, y, beta, rss_model, sqrt(var), var * inv, 1 - rss_model/rss, yhat, res,
    )
end

X(model::OLSModel) = model.X
y(model::OLSModel) = model.y
parameters( model::OLSModel ) = model.beta
RSS( model::OLSModel ) = model.rss

struct FTest
    test::Float64
    pvalue::Float64
end

function FTest( models::OLSModel{T,M,V}... ) where {T,M,V}
    @assert( length(models) == 2 )
    @assert( y(models[1]) == y(models[2]) )
    @assert( all( in.( eachcol(X(models[1])), [eachcol(X(models[2]))] ) ) )
    (p1, p2) = length.(parameters.(models))
    (RSS1, RSS2) = RSS.( models )

    df1 = p2 - p1
    df2 = length(y( models[1] )) - p2
    
    ftest = (RSS1 - RSS2)/df1/(RSS2/df2)
    return FTest( ftest, ccdf( FDist(df1, df2), ftest ) )
end

pvalue( ftest::FTest ) = ftest.pvalue

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

bound( x::AbstractVector{T}, buckets::Int ) where {T} = 
    quantile.( (x,), (d -> d:d:1-d)(1/buckets) )

label( x::AbstractVector{T}, bounds::Vector{T} ) where {T} = 
    searchsortedfirst.( (bounds,), x );

label( x::AbstractVector{T}, buckets::Int ) where {T} = label( x, bound( x, buckets ) )

function momentmap( x::AbstractVector{T}, y::AbstractVector{U}, buckets::Int; moments = 1  ) where {T,U}
#    (x,y) = makegood(x,y)
    n = length(x)

    bounds = bound( x, buckets )
    labels = label( x, bounds )

    counts = zeros( Int, buckets )
    xmoments = zeros( T, buckets )
    ymoments = zeros( U, buckets, moments )
    for i = 1:n
        l = labels[i]
        counts[l] += 1
        xmoments[l] += x[i]
        for j = 1:moments
            ymoments[l, j] += y[i]^j
        end
    end
    return Dict(
        :bounds => bounds,
        :counts => counts,
        :xmean => xmoments./counts,
        :ymoments => ymoments./counts,
    )
end

function multimomentmap( x::AbstractVector{T}, y::AbstractVector{U}, z::AbstractVector{V}, xbuckets::Int, zbuckets::Int; moments::Int = 1 ) where {T,U,V}
#    (x,y,z) = makegood(x,y,z)
    n = length(x)

    zbounds = bound( z, zbuckets )
    zlabels = label( z, zbounds )

    xbounds = bound( x, xbuckets )
    xlabels = label( x, xbounds )

    xmoments = zeros( T, zbuckets, xbuckets, moments + 1 )
    ymoments = zeros( U, zbuckets, xbuckets, moments + 1 )
    for i = 1:n
        lz = zlabels[i]
        lx = xlabels[i]
        for j = 0:moments
            xmoments[lz, lx, j+1] += x[i]^j
            ymoments[lz, lx, j+1] += y[i]^j
        end
    end
    for i in 1:zbuckets
        for j in 1:xbuckets
            for k = 2:moments+1
                xmoments[i,j,k] /= xmoments[i,j,1]
                ymoments[i,j,k] /= ymoments[i,j,1]
            end
            for k = moments+1:-1:3
                sumx = (-xmoments[i,j,2])^(k-1)
                sumy = (-ymoments[i,j,2])^(k-1)
                for l = 1:k-1
                    sumx += binomial( k-1, l ) * (-xmoments[i,j,2])^(k-1-l) * xmoments[i,j,l+1]
                    sumy += binomial( k-1, l ) * (-ymoments[i,j,2])^(k-1-l) * ymoments[i,j,l+1]
                end
                xmoments[i,j,k] = root( sumx, k-1 )
                ymoments[i,j,k] = root( sumy, k-1 )
            end
        end
    end
    return Dict(
        :zbounds => zbounds,
        :xbounds => xbounds,
        :xmoments => xmoments,
        :ymoments => ymoments,
    )
end

end # module AStat
