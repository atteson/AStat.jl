using QuadGK
using SpecialFunctions
using Distributions
using Random
using ReverseDiff
using LinearAlgebra
using Optim

normpdf( x, u, t ) = (sqrt(t)/sqrt(2*pi))^length(x) * exp(-t/2 * sum((x .- u).^2))

@assert( abs(pdf( Normal( 1, 2 ), 3 ) - normpdf( 3, 1, 1/4 )) < 1e-8 )
@assert( abs(prod(pdf( Normal( 1, 2 ), [1,2,3] )) - normpdf( [1,2,3], 1, 1/4 )) < 1e-8 )

f( x, t, u, a, b, u0, t0 ) = b^a/gamma(a) * t^(a-1) * exp(-b * t) * normpdf(u, u0, t0 * t) * normpdf(x, u, t)

function f1( x, t, u, a, b, u0, t0 )
    n = length(x)
    factor = b^a/gamma(a) * t^(n/2+a-1/2) * exp(-b * t) * sqrt(t0)/(2*pi)^((n+1)/2)
    term = t0 * (u - u0)^2 + sum((x .- u).^2)
    return factor * exp(-t/2 * term)
end

function f2( x, t, u, a, b, u0, t0 )
    n = length(x)
    factor = b^a/gamma(a) * t^(n/2+a-1/2) * exp(-b * t) * sqrt(t0)/(2*pi)^((n+1)/2)
    term = t0 * u^2 - 2 * t0 * u * u0 + t0 * u0^2 + sum(x.^2 .- 2 * u * x .+ u^2)
    return factor * exp(-t/2 * term)
end

function f3( x, t, u, a, b, u0, t0 )
    n = length(x)
    factor = b^a/gamma(a) * t^(n/2+a-1/2) * exp(-b * t) * sqrt(t0)/(2*pi)^((n+1)/2)
    term = (t0 + n) * u^2 - 2 * (t0 * u0 + sum(x)) * u + t0 * u0^2 + sum(x.^2)
    return factor * exp(-t/2 * term)
end

function f4( x, t, u, a, b, u0, t0 )
    n = length(x)
    factor = b^a/gamma(a) * t^(n/2+a-1/2) * exp(-b * t) * sqrt(t0)/(2*pi)^((n+1)/2)
    term = (t0 + n) * (u - (t0 * u0 + sum(x))/(t0 + n))^2 - (t0 * u0 + sum(x))^2/(t0 + n)
    term += t0 * u0^2 + sum(x.^2)
    return factor * exp(-t/2 * term)
end

function f( x, t, a, b, u0, t0 )
    n = length(x)
    xbar = mean(x)
    x2bar = mean(x.^2)
    factor = b^a/gamma(a) * sqrt(t0)/(sqrt(t0 + n) * (2 * pi)^(n/2)) * t^(n/2 + a - 1)
    return factor * exp(-t * (b + 1/2 * (t0 * u0^2 + n * x2bar - (t0 * u0 + n * xbar)^2/(t0 + n))))
end

function f( x, a, b, u0, t0 )
    n = length(x)
    factor = b^a/gamma(a) * sqrt(t0)/(sqrt(t0 + n) * (2 * pi)^(n/2))
    return factor * gamma(n/2 + a)/( b + 1/2 * (t0 * u0^2 + sum(x.^2) - (t0 * u0 + sum(x))^2/(t0 + n)))^(n/2 + a) 
end

x = collect(1.0:3.0)
t = 0.5
a = 2.0
b = 2.0
u0 = 0.0
t0 = 0.5

f( x, t, a, b, u0, t0 )

quadgk( u -> f( x, t, u, a, b, u0, t0 ), -Inf, Inf )
quadgk( u -> f1( x, t, u, a, b, u0, t0 ), -Inf, Inf )
quadgk( u -> f2( x, t, u, a, b, u0, t0 ), -Inf, Inf )
quadgk( u -> f3( x, t, u, a, b, u0, t0 ), -Inf, Inf )
quadgk( u -> f4( x, t, u, a, b, u0, t0 ), -Inf, Inf )

quadgk( t -> f( x, t, a, b, u0, t0 ), 0, Inf )

f( x, a, b, u0, t0 )

Random.seed!(1)
x = randn( 10 )
a = 1/rand()
b = 1/rand()
u0 = randn()
t0 = rand()

f( x, a, b, u0, t0 )

quadgk( t -> f( x, t, a, b, u0, t0 ), 0, Inf )

quadgk( x -> f( x, a, b, u0, t0 ), -Inf, Inf )

inputs = [a, b, u0, t0]
tape = ReverseDiff.GradientTape( v -> f( x, v[1], v[2], v[3], v[4] ), inputs )
compiled = ReverseDiff.compile( tape )
results = similar(inputs)
ReverseDiff.gradient!( results, compiled, inputs)

fidi = [(f( x, (inputs .+ 1e-6 * I(4)[i,:])... ) - f( x, (inputs .- 1e-6 * I(4)[i,:])... ))/2e-6 for i in 1:4]
@assert( all(abs.(fidi ./ results .- 1) .< 1e-6) )

n = Normal( u0, 1/sqrt(t0) )
@assert( all(abs.(pdf( n, -2:0.5:2 ) - normpdf.( -2:0.5:2, u0, t0 )) .< 1e-8) )

N = 1_000
Random.seed!(1)
t = rand( Gamma( a, b ), N );
u = rand.( Normal.( u0, 1 ./ sqrt.(t0 .* t) ) );
x = rand.( Normal.( u, 1 ./ sqrt.(t) ), 20 );

function g( x, a0, b0, u0, t0 )
    n = length(x)
    a = exp(a0)
    b = exp(b0)
    t = exp(t0)
    factor = b^a/gamma(a) * sqrt(t)/(sqrt(t + n) * (2 * pi)^(n/2))
    return factor * gamma(n/2 + a)/( b + 1/2 * (t * u0^2 + sum(x.^2) - (t * u0 + sum(x))^2/(t + n)))^(n/2 + a) 
end

inputs = zeros(4)
tape = ReverseDiff.GradientTape( v -> -sum(log.(g.( x, v[1], v[2], v[3], v[4] ))), inputs )
compiled = ReverseDiff.compile( tape )
results = similar(inputs)
@time ReverseDiff.gradient!( results, compiled, inputs);

fidi = [(-sum(log.(g.( x, (inputs .+ 1e-6 * I(4)[i,:])... ))) - -sum(log.(g.( x, (inputs .- 1e-6 * I(4)[i,:])... ))))/2e-6 for i in 1:4]
@assert( all(abs.(fidi ./ results .- 1) .< 1e-6) )

build_g( x ) = v -> -sum(log.(g.( x, v[1], v[2], v[3], v[4] )))

build_g( x )( inputs )

build_grad_g( x, tape ) = (grad, v) -> ReverseDiff.gradient!( grad, tape, v )

@assert( all( abs.(build_grad_g( x, compiled )(results, inputs) ./ fidi .- 1) .< 1e-7 ) )

@time solution = optimize( build_g( x ), build_grad_g( x, compiled ), inputs, ConjugateGradient() )
v = Optim.minimizer( solution )
[exp(v[1]) exp(v[2]) v[3] exp(v[4]); a b u0 t0]

