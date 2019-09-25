module BairdCounter

export BairdCounterEnv, reset!, observe, interact!

using ReinforcementLearningEnvironments
import ReinforcementLearningEnvironments: reset!, observe, interact!

const ACTIONS = (:dashed, :solid)

mutable struct BairdCounterEnv <: AbstractEnv
    current::Int
    observation_space::DiscreteSpace
    action_space::DiscreteSpace
    BairdCounterEnv() = new(rand(1:7), DiscreteSpace(7), DiscreteSpace(length(ACTIONS)))
end

function interact!(env::BairdCounterEnv, a)
    if ACTIONS[a] == :dashed
        env.current = rand(1:6)
    else
        env.current = 7
    end
    nothing
end

observe(env::BairdCounterEnv) =
    Observation(reward = 0.0, terminal = false, state = env.current)

function reset!(env::BairdCounterEnv)
    env.current = rand(1:6)
    nothing
end

end