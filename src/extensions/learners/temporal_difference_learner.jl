export TDLearner, DoubleLearner, DifferentialTDLearner, TDλReturnLearner

"""
    TDLearner(approximator::Tapp, γ::Float64, optimizer::Float64; n::Int=0) where {Tapp<:AbstractVApproximator}
    TDLearner(approximator::Tapp, γ::Float64, optimizer::Float64; n::Int=0, method::Symbol=:SARSA) where {Tapp<:AbstractQApproximator} 

The `TDLearner`(Temporal Difference Learner) use the latest `n` step experiences to update the `approximator`.

# Args

- `approximator`::[`AbstractApproximator`](@ref)
- Note that `n` starts with `0`, which means looking forward for the next `n` steps.
- `γ` is the discount rate of experience.
- `optimizer` is the learning rate.
- `method`. For [`VApproximator`](@ref), the only supported update method is `:SRS`, which means only **S**tates, **R**ewards and next_**S**ates are used to update the `approximator`. For [`QApproximator`](@ref), the following methods are supported:
    - `:SARS` (aka Q-Learning)
    - `:SARSA`
    - `:ExpectedSARSA`
"""
mutable struct TDLearner{Tapp<:AbstractApproximator,method,O} <: AbstractLearner
    approximator::Tapp
    γ::Float64
    optimizer::O
    n::Int

    function TDLearner(
        ;
        approximator::Tapp,
        γ = 1.0,
        optimizer = Descent(1.0),
        n = 0,
        method::Symbol = :SARSA,
    ) where {Tapp<:AbstractApproximator}
        if ApproximatorStyle(approximator) === VApproximator()
            if method != :SRS
                throw(ArgumentError("method [$method] is unsupported for a value approximator"))
            end
        elseif ApproximatorStyle(approximator) === QApproximator()
            if !in(method, [:SARSA, :SARS, :ExpectedSARSA])
                throw(ArgumentError("Supported methods are $supported_methods , your input is $method"))
            end
        else
            throw(ArgumentError("unknown approximator style"))
        end
        new{Tapp,method,typeof(optimizer)}(approximator, γ, optimizer, n)
    end
end

(learner::TDLearner)(obs) = learner.approximator(get_state(obs))
(learner::TDLearner)(obs, a) = learner.approximator(get_state(s), a)

RLBase.update!(learner::TDLearner, ::Nothing) = nothing
RLBase.update!(learner::TDLearner{T, M}, experience) where {T, M} = update!(learner, ApproximatorStyle(learner.approximator), Val(M), experience)

function RLBase.update!(
    learner::TDLearner,
    ::QApproximator,
    ::Val{:SARSA},
    transitions::NamedTuple{(:states, :actions, :rewards, :terminals, :next_states, :next_actions)}
)
    states, actions, rewards, terminals, next_states, next_actions = transitions
    n, γ, Q, optimizer = learner.n, learner.γ, learner.approximator, learner.optimizer

    if length(terminals) > 0 && terminals[end]
        @views gains = discount_rewards(rewards[max(end - n, 1):end], γ)  # n starts with 0
        for (i, G) in enumerate(gains)
            @views s, a = states[end-length(gains)+i], actions[end-length(gains)+i]
            update!(Q, (s, a) => apply!(optimizer, (s, a), G - Q(s, a)))
        end
    else
        if length(states) ≥ (n + 1)  # n starts with 0
            @views s, a, s′, a′ = states[end-n],
                actions[end-n],
                next_states[end],
                next_actions[end]
            @views G = discount_rewards_reduced(rewards[end-n:end], γ) +
                       γ^(n + 1) * Q(s′, a′)
            update!(Q, (s, a) => apply!(optimizer, (s, a), G - Q(s, a)))
        end
    end
end

RLBase.extract_experience(t::AbstractTrajectory, learner::TDLearner{T, M}) where {T, M} = extract_experience(t, learner, ApproximatorStyle(learner.approximator), Val(M))

function RLBase.extract_experience(
    t::AbstractTrajectory,
    learner::TDLearner,
    ::QApproximator,
    ::Val{:SARSA},
)
    n, N = learner.n, length(t)
    # !!! n starts with 0
    if N > 0
        (
            states = select_last_dim(get_trace(t, :state), N-n:N),
            actions = select_last_dim(get_trace(t, :action), N-n:N),
            rewards = select_last_dim(get_trace(t, :reward), N-n:N),
            terminals = select_last_dim(get_trace(t, :terminal), N-n:N),
            next_states = select_last_dim(get_trace(t, :next_state), N-n:N),
            next_actions = select_last_dim(get_trace(t, :next_action), N-n:N),
        )
    else
        nothing
    end
end