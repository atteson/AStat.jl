using AStat
using Distributions
using Random

μ₀ = Constant()
σ₀ = open_interval( 0, Inf )
X = IndependentRandomVariable( Normal( μ₀, σ₀ ) )

@assert( Random.gentype(X) == Float64 )

μ₁ = open_interval( -Inf32, Inf32 )
σ₁ = open_interval( 0f0, Inf32 )
Y = IndependentRandomVariable( Normal( μ₁, σ₁ ) )

@assert( Random.gentype(Y) == Float32 )

rand( X, 20 )

import Random: default_rng, gentype, Sampler, typeof_rng
@which rand( X, 10 )
@which rand(X, Dims(10,))
@which rand(default_rng(), X, Dims(10,))
rng = r = default_rng()
dims = Dims(10,)
gentype(X)
@which rand!(r, Array{gentype(X)}(undef, dims), X)
A = Array{gentype(X)}(undef, dims)
sp = Sampler(rng, X)
@which rand!(rng, A, sp)
@which rand(rng, sp)

x = X
r=Val(Inf)
Sampler(typeof_rng(rng), x, r)

rand!(rng, A, sp)
