using AStat
using Distributions
using Random

μ₀ = Constant()
σ₀ = open_interval( 0, Inf )
X = IndependentRandomVariable( Normal, μ₀, σ₀ )

@assert( Random.gentype(X) == Float64 )

μ₁ = open_interval( -Inf32, Inf32 )
σ₁ = open_interval( 0f0, Inf32 )
Y = IndependentRandomVariable( Normal, μ₁, σ₁ )

@assert( Random.gentype(Y) == Float32 )

μ₀[] = 0.0
σ₀[] = 1.0
rand( X, 20 )

rand( X, 20, 10 )

Z = IndependentRandomVariable( Normal, X, σ₀ )
@time std( rand( Z, 1_000_000 ) )
@enter( rand( Z, 10 )
