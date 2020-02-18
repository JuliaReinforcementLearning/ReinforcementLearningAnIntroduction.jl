@reexport module BairdCounter

export BairdCounterEnv

using ReinforcementLearningBase

const ACTIONS = (:dashed, :solid)

Base.@kwdef mutable struct BairdCounterEnv <: AbstractEnv
    current::Int = rand(1:7)
end

RLBase.get_observation_space(env::BairdCounterEnv) = DiscreteSpace(7)
RLBase.get_action_space(env::BairdCounterEnv) = DiscreteSpace(length(ACTIONS))

function (env::BairdCounterEnv)(a)
    if ACTIONS[a] == :dashed
        env.current = rand(1:6)
    else
        env.current = 7
    end
    nothing
end

RLBase.observe(env::BairdCounterEnv) =
    (reward = 0.0, terminal = false, state = env.current)

function RLBase.reset!(env::BairdCounterEnv)
    env.current = rand(1:6)
    nothing
end

end