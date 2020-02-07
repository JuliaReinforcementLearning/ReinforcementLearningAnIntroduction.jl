export MonteCarloLearner,
       FIRST_VISIT,
       EVERY_VISIT,
       NO_SAMPLING,
       ORDINARY_IMPORTANCE_SAMPLING,
       WEIGHTED_IMPORTANCE_SAMPLING

using StatsBase: countmap

abstract type AbstractVisitType end
struct FirstVisit <: AbstractVisitType end
struct EveryVisit <: AbstractVisitType end
const FIRST_VISIT = FirstVisit()
const EVERY_VISIT = EveryVisit()

abstract type AbstractSamplingStyle end
struct NoSampling <: AbstractSamplingStyle end
struct OrdinaryImportanceSampling <: AbstractSamplingStyle end
struct WeightedImportanceSampling <: AbstractSamplingStyle end
const NO_SAMPLING = NoSampling()
const ORDINARY_IMPORTANCE_SAMPLING = OrdinaryImportanceSampling()
const WEIGHTED_IMPORTANCE_SAMPLING = WeightedImportanceSampling()

"""
    MonteCarloLearner(;kwargs...)

# Fields

- `approximator`::[`AbstractApproximator`](@ref),
"""
struct MonteCarloLearner{T,A<:AbstractApproximator,R,S} <: AbstractLearner
    approximator::A
    γ::Float64
    α::Float64
    returns::R

    MonteCarloLearner(
        ;
        approximator::A,
        γ = 1.0,
        α = 1.0,
        kind = FIRST_VISIT,
        sampling = NO_SAMPLING,
        returns = CachedSampleAvg{Float64}(),
    ) where {A} =
        new{typeof(kind),A,typeof(returns),typeof(sampling)}(approximator, γ, α, returns)
end

VisitStyle(::MonteCarloLearner{T}) where T = T
SamplingStyle(::MonteCarloLearner{T,A,R,S}) where {T,A,R,S} = S

RLBase.ApproximatorStyle(m::MonteCarloLearner) = ApproximatorStyle(m.approximator)

(learner::MonteCarloLearner)(obs) = learner.approximator(get_state(obs))
(learner::MonteCarloLearner)(obs, a) = learner.approximator(get_state(s), a)

RLBase.update!(learner::MonteCarloLearner, experience) = update!(learner, VisitStyle(learner), ApproximatorStyle(learner), SamplingStyle(learner), experience)

RLBase.extract_experience(t::AbstractTrajectory, learner::MonteCarloLearner) = extract_experience(t, learner, ApproximatorStyle(learner))

function RLBase.extract_experience(t::AbstractTrajectory, learner::MonteCarloLearner, ::VApproximator)
    # only extract & update at the end of an episode
    if length(t) > 0 && get_trace(t, :terminal)[end]
        get_trace(t, :state, :reward)
    else
        nothing
    end
end

function RLBase.extract_experience(t::AbstractTrajectory, learner::MonteCarloLearner, ::QApproximator)
    # only extract & update at the end of an episode
    if length(t) > 0 && get_trace(t, :terminal)[end]
        get_trace(t, :state, :action, :reward)
    else
        nothing
    end
end

RLBase.update!(learner::MonteCarloLearner, ::Any, ::Any, ::Any, ::Nothing) = nothing

function RLBase.update!(
    learner::MonteCarloLearner,
    ::Type{FirstVisit},
    ::VApproximator,
    ::Type{NoSampling},
    transitions::NamedTuple{(:state, :reward)},
)
    states, rewards = transitions.state, transitions.reward
    V, γ, α, Returns, G, T = learner.approximator,
        learner.γ,
        learner.α,
        learner.returns,
        0.0,
        length(states)
    seen_states = countmap(states)

    for t = T:-1:1
        S, R = states[t], rewards[t]
        G = γ * G + R
        if seen_states[S] == 1  # first visit
            update!(V, S => α * (Returns(S, G) - V(S)))
            delete!(seen_states, S)
        else
            seen_states[S] -= 1
        end
    end
end

function RLBase.update!(
    learner::MonteCarloLearner,
    ::Type{EveryVisit},
    ::VApproximator,
    ::Type{NoSampling},
    transitions::NamedTuple{(:state, :reward)},
)
    states, rewards = transitions.state, transitions.reward
    α, γ, V, Returns, G = learner.α, learner.γ, learner.approximator, learner.returns, 0.0
    for (s, r) in Iterators.reverse(zip(states, rewards))
        G = γ * G + r
        update!(V, s => α * (Returns(s, G) - V(s)))
    end
end