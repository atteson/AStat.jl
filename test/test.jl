using AStat
using Random

n = 100_000_000
x = randperm( n );
@time s = lis(x);
@assert( issorted(x[s]) )

@assert( abs(length(s) - 2*sqrt(n) + 1.77108 * n^(1/6)) < n^(1/6) )
