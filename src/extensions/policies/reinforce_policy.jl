export ReinforcePolicy

using Flux: softmax
using LinearAlgebra: dot
using StatsBase

"""
    ReinforcePolicy(approximator, α, γ)
This is the very basic reinforce algorithm.
TODO: implement some other variants.
"""
Base.@kwdef struct ReinforcePolicy{A<:AbstractApproximator} <: AbstractPolicy
    approximator::A
    α::Float64
    γ::Float64
end

(π::ReinforcePolicy)(obs) =
    obs |> get_state |> π.approximator |> softmax |> x -> Weights(x, 1.0) |> sample

RLBase.get_prob(π::ReinforcePolicy, s) = s |> π.approximator |> softmax
RLBase.get_prob(π::ReinforcePolicy, s, a) = get_prob(π, s)[a]

# TODO: handle neural network q approximator
function RLBase.update!(π::ReinforcePolicy{<:LinearQApproximator}, t::AbstractTrajectory)
    if length(t) > 0 && get_trace(t, :terminal)[end]
        states, actions, rewards = get_trace(t, :state), get_trace(t, :action), get_trace(t, :reward)
        Q, α, γ = π.approximator, π.α, π.γ
        gains = discount_rewards(rewards, γ)
        γₜ = 1.0

        for (i, (s, a, g)) in enumerate(zip(states, actions, gains))
            # !!! we will multiply `Q.feature_func(s, a)` in the `LinearQApproximator` again!
            update!(
                Q,
                (s, a) => α * γₜ * g *
                          (Q.feature_func(s, a) .-
                           sum(x -> get_prob(π, s, x) .* Q.feature_func(s, x), Q.actions)),
            )
            γₜ *= γ
        end
    end
end