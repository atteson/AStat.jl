
export IndependentRandomVariable, Constant, open_interval

using Distributions
using Random

abstract type RandomVariable <: Real
end

struct IndependentRandomVariable{T <: Distribution} <: RandomVariable
    distribution::T
end

struct Constant{T} <: RandomVariable
    lo::Vector{T}
    left::Vector{Bool}
    hi::Vector{T}
    right::Vector{Bool}
end

Constant() = Constant( [-Inf], [false], [Inf], [false] )
Constant{T}( x ) where T = ( x -> Constant( [x], [true], [x], [true] ) )(convert( T, x ))
open_interval( x, y ) = ( (x, y) -> Constant( [x], [false], [y], [false] ) )(promote(x,y)...)
Random.gentype( X::IndependentRandomVariable{T} ) where T = promote_type(Random.gentype.( params( X.distribution ) )...)
Random.gentype( c::Constant{T} ) where T = T

Base.:<( x::Constant{T}, y::Constant{T} ) where T = x.hi[end] < y.lo[end] || ( x.hi[end] == y.lo[end] && !( x.right[end] && y.left[end] ) )

function Base.push!( x::Constant{T}, y::U ) where {T, U <: Number}
    @assert( ( y < x.hi[end] || ( y == x.hi[end] && x.right[end] ) ) && ( y > x.lo[end] || ( y == x.lo[end] && x.left[end] ) ) )
    push!( x.lo, y )
    push!( x.left, true )
    push!( x.hi, y )
    push!( x.right, true )
    return x
end

Base.pop!( x::Constant{T} ) where T = (pop!( x.lo ), pop!( x.left ), pop!( x.hi ), pop!( x.right ))

function generate_distribution( rng::AbstractRNG, distribution::Distribution;
                                env::Dict{RandomVariable,AbstractArray{T}} = Dict{RandomVariable,AbstractArray{T}}() ) where T
    generated_parmas = T[]
    for param in params(distribution)
        if !haskey( env, param )
            env[param] = rand!(rng, Random.SamplerTrivial(param), env=env )
        end
        push!( generated_params, env[param] )
    end
    return Base.typename(U).wrapper(generated_params...)
end

function Random.rand!(rng::AbstractRNG, A::AbstractArray{T}, sp::Random.SamplerTrivial{V};
                      env::Dict{RandomVariable,AbstractArray{T}} = Dict{RandomVariable,AbstractArray{T}}() ) where
    {T, U, V <: Constant{U}}
    
    c = sp.self
    @assert( c.lo == c.hi && c.left && c.right )
    return c.lo
end

function Random.rand!(rng::AbstractRNG, A::AbstractArray{T}, sp::Random.SamplerTrivial{V};
                      env::Dict{RandomVariable,AbstractArray{T}} = Dict{RandomVariable,AbstractArray{T}}() ) where
    {T, U, V <: IndependentRandomVariable{U}}

                      
    for index in eachindex(A)
        A[index] = rand!( rng, generate_distribution( rng, sp.self.distribution, env=env ) )
    end
    return A
end

struct IIDRandomProcess{T <: Distribution}
    distribution::T
end

function Random.rand!(rng::AbstractRNG, A::AbstractArray{T}, sp::Random.SamplerTrivial{V};
                      env::Dict{RandomVariable,AbstractArray{T}} = Dict{RandomVariable,AbstractArray{T}}() ) where
    {T, U, V <: IIDRandomProcess{U}}

    (n, s) = Iterators.peel(size(A))
    for i in CartesianIndices(([1:x for x in s]...,))
        A[:,i] = rand!( rng, generate_distribution( rng, sp.self.distribution, env=env ), n )
    end
    return A
end
