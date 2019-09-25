module MaximizationBias

export MaximizationBiasEnv, reset!, observe, interact!

using ReinforcementLearningEnvironments
import ReinforcementLearningEnvironments: reset!, observe, interact!

"""
states:
1:A
2:B
3:terminal

actions:
1: left
2: right
"""
mutable struct MaximizationBiasEnv <: AbstractEnv
    position::Int
    reward::Float64
    observation_space::DiscreteSpace
    action_space::DiscreteSpace

    MaximizationBiasEnv() = new(1, 0.0, DiscreteSpace(3), DiscreteSpace(10))
end

const LEFT = 1

function interact!(env::MaximizationBiasEnv, a::Int)
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

function reset!(env::MaximizationBiasEnv)
    env.position = 1
    env.reward = 0.0
    nothing
end

observe(env::MaximizationBiasEnv) =
    Observation(reward = env.reward, terminal = env.position == 3, state = env.position)

end