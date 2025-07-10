using AStat
using Random
using Statistics

Random.seed!( 1 )

n = 100
m = 10
N = 10_000
X = randn( n, m );
y = randn( n, N );

@time olses = OLSModel.( [X], eachcol(y) );
X0 = Matrix{Float64}( undef, n, 0 )
null_olses = OLSModel.( [X0], eachcol(y) );
@time ftests = FTest.( null_olses, olses );
@time pvalues = pvalue.( ftests );

@assert( abs(mean( pvalues ) - 0.5) < 0.005 )
@assert( abs(std( pvalues ) - sqrt( 1/12 )) < 0.005 )
