
export IndependentRandomVariable, Constant, open_interval

using Distributions
using Random
using MissingTypes

# subtype of Real so that they can be used as parameters of Distributions
abstract type RandomVariable <: Real
end

mutable struct Constant{T} <: RandomVariable
    lo::T
    left::Bool
    hi::T
    right::Bool
    value::Vector{MissingType{T}}
    visited::Bool
end

mutable struct IndependentRandomVariable{T <: Distribution, U <: Random.gentype(Random.gentype(T))} <: RandomVariable
    distribution::T
    value::Vector{MissingType{U}}
    visited::Bool
end

Constant() = Constant{Float64}( -Inf, false, Inf, false, MissingType{Float64}[MissingTypes.missing_value(Float64)], false )
# needed for constructing Normal
Constant{T}( x ) where T = ( x -> Constant{T}( x, true, x, true, convert( Vector{MissingType{T}}, [x] ), false ) )( convert( T, x ) )
function open_interval( x, y )
    (x,y) = promote(x,y)
    T = typeof(x)
    return Constant{T}( x, false, y, false, [MissingTypes.missing_value(T)], false )
end

function IndependentRandomVariable( distribution::Distribution )
    T = Random.gentype(Random.gentype(distribution))
    return IndependentRandomVariable( distribution, MissingType{T}[MissingTypes.missing_value(T)], false )
end

Random.gentype( c::Constant{T} ) where T = T
Random.gentype( ::Type{Constant{T}} ) where T = T
Random.gentype( X::IndependentRandomVariable{T} ) where T = promote_type(Random.gentype.( params( X.distribution ) )...)

Base.:<( x::Constant{T}, y::Constant{T} ) where T = x.hi < y.lo || ( x.hi == y.lo && !( x.right && y.left ) )

function Base.setindex!( x::Constant{T}, y::U ) where {T, U <: Number}
    @assert( ( y < x.hi || ( y == x.hi && x.right ) ) && ( y > x.lo || ( y == x.lo && x.left ) ) )
    x.value[1] = y
    return x
end

function clear!( x::Constant{T} ) where T
    x.visited = false
end

function clear!( x::IndependentRandomVariable{T} ) where T
    if x.visited
        x.visited = false
        for param in params(distribution)
            clear!( param )
        end
    end
end

function generate_distribution( rng::AbstractRNG, distribution::U ) where {U <: Distribution}
    generated_params = Random.gentype(Random.gentype(distribution))[]
    for param in params(distribution)
        if !param.visited
            param.visited = true
            rand!(rng, param.value, Random.SamplerTrivial(param), clear=false )
        end
        push!( generated_params, param.value[1] )
    end
    return Base.typename(U).wrapper(generated_params...)
end

function Random.rand!(rng::AbstractRNG, A::AbstractArray{T}, sp::Random.SamplerTrivial{V}; clear::Bool = true ) where {T,V}
    @assert( !ismissing( sp.self.value[1] ) )
    A .= sp.self.value[1]
    if clear
        clear!( sp.self )
    end
    return A
end

function Random.rand!(rng::AbstractRNG, A::AbstractArray{T}, sp::Random.SamplerTrivial{V};
                      clear::Bool = true ) where{T, U, V <: IndependentRandomVariable{U}}
    for index in eachindex(A)
        A[index] = rand( rng, generate_distribution( rng, sp.self.distribution ) )
    end
    if clear
        clear!( sp.self )
    end
    return A
end

struct IIDRandomProcess{T <: Distribution}
    distribution::T
end

function Random.rand!(rng::AbstractRNG, A::AbstractArray{T}, sp::Random.SamplerTrivial{IIDRandomProcess{U}} ) where {T, U}
    (n, s) = Iterators.peel(size(A))
    for i in CartesianIndices(([1:x for x in s]...,))
        A[:,i] = rand!( rng, generate_distribution( rng, sp.self.distribution ), n )
    end
    return A
end
