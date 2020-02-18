export TabularRandomPolicy

using StatsBase:Weights, sample

"""
    TabularRandomPolicy(prob::Array{Float64, 2})

`prob` describes the distribution of actions for each state.
"""
struct TabularRandomPolicy <: AbstractPolicy
    prob::Array{Float64,2}
end

(π::TabularRandomPolicy)(s::Int) = sample(Weights(get_prob(π, s)))
(π::TabularRandomPolicy)(obs) = π(get_state(obs))

RLBase.get_prob(π::TabularRandomPolicy, s) = @view π.prob[:, s]
RLBase.get_prob(π::TabularRandomPolicy, s, a) = π.prob[a, s]