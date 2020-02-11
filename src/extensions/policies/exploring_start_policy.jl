export ExploringStartPolicy

using Random
using MacroTools: @forward

mutable struct ExploringStartPolicy{P<:AbstractPolicy, A, R<:AbstractRNG} <: AbstractPolicy
    policy::P
    actions::A
    is_start::Bool
    rng::R
end

ExploringStartPolicy(;policy, actions, is_start=true,seed=nothing) = ExploringStartPolicy(policy, actions, is_start, MersenneTwister(seed))

@forward ExploringStartPolicy.policy RLBase.get_prob, RLBase.update!

(p::ExploringStartPolicy)(obs) = p.is_start ? rand(p.rng, p.actions) : p.policy(obs)

function (agent::Agent{<:ExploringStartPolicy,<:EpisodicCompactSARTSATrajectory})(
    ::PreEpisodeStage,
    obs,
)
    empty!(agent.trajectory)
    agent.policy.is_start = true
    nothing
end

function (agent::Agent{<:ExploringStartPolicy,<:EpisodicCompactSARTSATrajectory})(
    ::PostActStage,
    obs,
)
    push!(agent.trajectory; reward = get_reward(obs), terminal = get_terminal(obs))
    agent.policy.is_start = false
    nothing
end