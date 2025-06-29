using AStat
using Random
using Statistics

Random.seed!( 1 )

n = 100
m = 10
N = 100_000
X = randn( n, m );
y = randn( n, N );

@time olses = ols.( [X], eachcol(y) );
pvalues = getindex.( olses, :pvalue );
@assert( abs(mean( pvalues ) - 0.5) < 0.005 )
@assert( abs(std( pvalues ) - sqrt( 1/12 )) < 0.005 )
