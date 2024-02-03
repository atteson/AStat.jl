
export IndependentRandomVariable, Constant, open_interval, IIDRandomProcess

using Distributions
using Random
import Random: gentype
using MissingTypes

abstract type RandomElement{T}
end

gentype( ::RandomElement{T} ) where T = T

mutable struct Constant{T} <: RandomElement{T}
    lo::T
    left::Bool
    hi::T
    right::Bool
    value::Vector{MissingType{T}}
    visited::Bool
end

mutable struct IndependentRandomVariable{T <: Distribution, U} <: RandomElement{U}
    parameters::Vector{RandomElement}
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

function IndependentRandomVariable( ::Type{T}, parameters::RandomElement... ) where {T <: Distribution}
    U = promote_type( gentype.( parameters )... )
    return IndependentRandomVariable{T,U}( [parameters...], MissingType{U}[MissingTypes.missing_value(U)], false )
end

function Base.setindex!( x::Constant{T}, y::U ) where {T, U <: Number}
    @assert( ( y < x.hi || ( y == x.hi && x.right ) ) && ( y > x.lo || ( y == x.lo && x.left ) ) )
    x.value[1] = y
    return x
end

function clear!( x::Constant{T} ) where T
end

function Random.rand!(rng::AbstractRNG, A::AbstractArray{U}, sp::Random.SamplerTrivial{Constant{T}};
                      clear::Bool = true ) where {T, U <: Union{T,MissingType{T}}}
    @assert( !ismissing( sp.self.value[1] ) )
    sp.self.visited = true
    A .= sp.self.value[1]
    if clear
        clear!( sp.self )
    end
    return A
end

function Random.rand!(rng::AbstractRNG, A::AbstractArray{V}, sp::Random.SamplerTrivial{IndependentRandomVariable{T,U}};
                      clear::Bool = true ) where{T, U, V <: Union{U,MissingType{U}}}
    for index in eachindex(A)
        sp.self.visited = true
        generated_parameters = U[]
        for parameter in sp.self.parameters
            if !parameter.visited
                rand!( rng, parameter.value, Random.SamplerTrivial(parameter), clear=false )
            end
            push!( generated_parameters, parameter.value[1] )
        end
        A[index] = rand( rng, T( generated_parameters... ) )
        if clear
            clear!( sp.self )
        end
    end
    return A
end

abstract type AbstractSequence{T}
end

mutable struct IIDRandomProcess{T, U} <: RandomElement{AbstractSequence{U}}
    parameters::Vector{RandomElement}
    visited::Bool
end

function IIDRandomProcess( ::Type{T}, parameters::RandomElement... ) where {T <: Distribution}
    U = promote_type( gentype.( parameters )... )
    return IIDRandomProcess{T,U}( [parameters...], false )
end

gentype( ::IIDRandomProcess{T,U} ) where {T,U} = U

function clear!( x::Union{IndependentRandomVariable{T,U}, IIDRandomProcess{T,U}} ) where {T,U}
    if x.visited
        x.visited = false
        for parameter in x.parameters
            clear!( parameter )
        end
    end
end

function Random.rand!(rng::AbstractRNG, A::AbstractArray{V}, sp::Random.SamplerTrivial{IIDRandomProcess{T,U}};
                      clear::Bool = true ) where{T, U, V <: Union{U,MissingType{U}}}
    (n, s) = Iterators.peel(size(A))
    for i in CartesianIndices(([1:x for x in s]...,))
        sp.self.visited = true
        generated_parameters = U[]
        for parameter in sp.self.parameters
            if !parameter.visited
                rand!( rng, parameter.value, Random.SamplerTrivial(parameter), clear=false )
            end
            push!( generated_parameters, parameter.value[1] )
        end
        A[:,i] = rand( rng, T( generated_parameters... ), n )
        if clear
            clear!( sp.self )
        end
    end
    return A
end
