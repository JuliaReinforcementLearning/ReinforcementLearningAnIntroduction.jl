@reexport module MaximizationBias

export MaximizationBiasEnv

using ReinforcementLearningBase


"""
states:
1:A
2:B
3:terminal

actions:
1: left
2: right
"""
Base.@kwdef mutable struct MaximizationBiasEnv <: AbstractEnv
    position::Int = 1
    reward::Float64 = 0.0
end

RLBase.get_observation_space(env::MaximizationBiasEnv) = DiscreteSpace(3)
RLBase.get_action_space(env::MaximizationBiasEnv) = DiscreteSpace(10)

const LEFT = 1

function (env::MaximizationBiasEnv)(a::Int)
    if env.position == 1
        if a == 1
            env.position = 2
            env.reward = 0.0
        else
            env.position = 3
            env.reward = 0.0
        end
    elseif env.position == 2
        env.position = 3
        env.reward = randn() - 0.1
    end
    nothing
end

function RLBase.reset!(env::MaximizationBiasEnv)
    env.position = 1
    env.reward = 0.0
    nothing
end

RLBase.observe(env::MaximizationBiasEnv) =
    (reward = env.reward, terminal = env.position == 3, state = env.position)

end