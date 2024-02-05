using QuadGK
using SpecialFunctions
using Distributions

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
    factor = b^a/gamma(a) * sqrt(t0)/(sqrt(t0 + n) * (2*pi)^(n/2)) * t^(n/2 + a - 1)
    return factor * exp(-t * (b + 1/2*(t0 * u0^2 + n * x2bar - (t0 * u0 + n * xbar)^2/(t0 + n))))
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

dt = 0.0001
sum((u -> f( x, t, u, a, b, u0, t0 )).(-20:dt:20))*dt
sum((u -> f1( x, t, u, a, b, u0, t0 )).(-20:dt:20))*dt
