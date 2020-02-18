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

(learner::MonteCarloLearner)(obs) = learner.approximator(obs)
(learner::MonteCarloLearner)(obs, a) = learner.approximator(s, a)

RLBase.update!(learner::MonteCarloLearner, experience) = update!(learner, VisitStyle(learner), ApproximatorStyle(learner), SamplingStyle(learner), experience)

function RLBase.extract_experience(t::AbstractTrajectory, learner::MonteCarloLearner)
    # only extract & update at the end of an episode
    if length(t) > 0 && get_trace(t, :terminal)[end]
        (
            states=get_trace(t, :state),
            actions = get_trace(t, :action),
            rewards = get_trace(t, :reward),
        )
    else
        nothing
    end
end

function RLBase.update!(
    learner::MonteCarloLearner,
    ::Type{FirstVisit},
    ::VApproximator,
    ::Type{NoSampling},
    transitions::NamedTuple{(:states, :actions, :rewards)},
)
    states, rewards = transitions.states, transitions.rewards
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
    ::Type{FirstVisit},
    ::VApproximator,
    ::Type{OrdinaryImportanceSampling},
    transitions::NamedTuple{(:states, :actions, :rewards, :weights)},
)
    states, rewards, weights = transitions.states, transitions.rewards, transitions.weights
    V, γ, α, Returns, G, ρ, T = learner.approximator,
        learner.γ,
        learner.α,
        learner.returns,
        0.0,
        1.0,
        length(states)
    seen_states = countmap(states)

    for t = T:-1:1
        S, R = states[t], rewards[t]
        G = γ * G + R
        ρ *= weights[t]
        if seen_states[S] == 1  # first visit
            update!(V, S => α * (Returns(S, ρ * G) - V(S)))
            delete!(seen_states, S)
        else
            seen_states[S] -= 1
        end
    end
end

function RLBase.update!(
    learner::MonteCarloLearner,
    ::Type{FirstVisit},
    ::VApproximator,
    ::Type{WeightedImportanceSampling},
    transitions::NamedTuple{(:states, :actions, :rewards, :weights)},
)
    states, rewards, weights = transitions.states, transitions.rewards, transitions.weights
    V, γ, α, (G_cached, ρ_cached), G, ρ, T = learner.approximator,
        learner.γ,
        learner.α,
        learner.returns,
        0.0,
        1.0,
        length(states)
    seen_states = countmap(states)

    for t = T:-1:1
        S, R = states[t], rewards[t]
        G = γ * G + R
        ρ *= weights[t]
        if seen_states[S] == 1  # first visit
            numerator = G_cached(S, ρ * G)
            denominator = ρ_cached(S, ρ)
            val = denominator == 0 ? 0 : numerator / denominator
            update!(V, S => α * (val - V(S)))
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
    transitions::NamedTuple{(:states, :actions, :rewards)},
)
    states, rewards = transitions.states, transitions.rewards
    α, γ, V, Returns, G = learner.α, learner.γ, learner.approximator, learner.returns, 0.0
    for (s, r) in Iterators.reverse(zip(states, rewards))
        G = γ * G + r
        update!(V, s => α * (Returns(s, G) - V(s)))
    end
end

function RLBase.update!(
    learner::MonteCarloLearner,
    ::Type{FirstVisit},
    ::QApproximator,
    ::Type{NoSampling},
    transitions::NamedTuple{(:states, :actions, :rewards)},
)
    states, actions, rewards = transitions
    α, γ, Q, Returns, G, T = learner.α,
        learner.γ,
        learner.approximator,
        learner.returns,
        0.0,
        length(states)
    seen_pairs = countmap(zip(states, actions))

    for t = T:-1:1
        S, A, R = states[t], actions[t], rewards[t]
        pair = (S, A)
        G = γ * G + R
        if seen_pairs[pair] == 1  # first visit
            update!(Q, pair => α * (Returns(pair, G) - Q(S, A)))
            delete!(seen_pairs, pair)
        else
            seen_pairs[pair] -= 1
        end
    end
end

function RLBase.update!(
    learner::MonteCarloLearner,
    ::Type{EveryVisit},
    ::QApproximator,
    ::Type{NoSampling},
    transitions::NamedTuple{(:states, :actions, :rewards)},
)
    states, actions, rewards = transitions
    α, γ, Q, Returns, G = learner.α, learner.γ, learner.approximator, learner.returns, 0.0
    for (s, a, r) in Iterators.reverse(zip(states, actions, rewards))
        G = γ * G + r
        update!(Q, (s, a) => α * (Returns((s, a), G) - Q(s, a)))
    end
end