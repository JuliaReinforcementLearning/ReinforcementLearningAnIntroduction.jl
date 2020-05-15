export LinearVApproximator, LinearQApproximator

using LinearAlgebra: dot

"""
    LinearVApproximator(weights::Array{Float64, N}) -> LinearVApproximator{N}
Use the weighted sum to represent the estimation of a state.
The state is expected to have the same length with `weights`.
See also [`LinearQApproximator`](@ref)
"""
struct LinearVApproximator{N} <: AbstractApproximator
    weights::Array{Float64,N}
end

RLCore.ApproximatorStyle(::LinearVApproximator) = V_APPROXIMATOR

# TODO: support Vector
(V::LinearVApproximator)(s) = dot(s, V.weights)

function RLBase.update!(V::LinearVApproximator, correction::Pair)
    s, e = correction
    V.weights .+= s .* e
end

"""
    LinearQApproximator(weights::Vector{Float64}, feature_func::F, actions::Vector{Int}) -> LinearQApproximator{F}
Use weighted sum to represent the estimation given a state and an action.
# Fields
- `weights::Vector{Float64}`: the weight of each feature.
- `feature_func::Function`: decide how to generate a feature vector of `length(weights)` given a state and an action as parameters.
- `actions::Vector{Int}`: all possible actions.
See also [`LinearVApproximator`](@ref).
"""
Base.@kwdef struct LinearQApproximator{F} <: AbstractApproximator
    weights::Vector{Float64}
    feature_func::F
    actions::Vector{Int}
end

RLCore.ApproximatorStyle(::LinearQApproximator) = Q_APPROXIMATOR

(Q::LinearQApproximator)(s, a::Int) = dot(Q.weights, Q.feature_func(s, a))

(Q::LinearQApproximator)(s) = [Q(s, a) for a in Q.actions]

function RLBase.update!(Q::LinearQApproximator, correction::Pair)
    (s, a), e = correction
    xs = Q.feature_func(s, a)
    Q.weights .+= xs .* e
end