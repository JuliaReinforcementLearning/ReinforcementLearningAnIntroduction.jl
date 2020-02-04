module LeftRight

export LeftRightEnv

using ReinforcementLearningCore


using StatsBase

mutable struct LeftRightEnv <: AbstractEnv
    transitions::Array{Float64,3}
    current_state::Int
end

RLBase.get_observation_space(env::LeftRightEnv) = DiscreteSpace(2)
RLBase.get_action_space(env::LeftRightEnv) = DiscreteSpace(2)

function LeftRightEnv()
    t = zeros(2, 2, 2)
    t[1, :, :] = [0.9 0.1; 0.0 1.0]
    t[2, :, :] = [0.0 1.0; 0.0 1.0]
    LeftRightEnv(t, rand(1:2))
end

function (env::LeftRightEnv)(a::Int)
    env.current_state = sample(Weights(
        @view(env.transitions[env.current_state, a, :]),
        1.0,
    ))
    nothing
end

function RLBase.reset!(env::LeftRightEnv)
    env.current_state = 1
    nothing
end

RLBase.observe(env::LeftRightEnv) =
    (
        reward = Float64(env.current_state == 2),
        terminal = env.current_state == 2,
        state = env.current_state,
    )

end