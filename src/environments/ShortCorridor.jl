module ShortCorridor

export ShortCorridorEnv

using ReinforcementLearningCore


Base.@kwdef mutable struct ShortCorridorEnv <: AbstractEnv
    position::Int = 1
end

RLBase.get_observation_space(env::ShortCorridorEnv) = DiscreteSpace(4)
RLBase.get_action_space(env::ShortCorridorEnv) = DiscreteSpace(2)

function (env::ShortCorridorEnv)(a)
    if env.position == 1 && a == 2
        env.position += 1
    elseif env.position == 2
        env.position += a == 1 ? 1 : -1
    elseif env.position == 3
        env.position += a == 1 ? -1 : 1
    end
    nothing
end

function RLBase.reset!(env::ShortCorridorEnv)
    env.position = 1
    nothing
end

RLBase.observe(env::ShortCorridorEnv) =
    (
        state = env.position,
        terminal = env.position == 4,
        reward = env.position == 4 ? 0.0 : -1.0,
    )

end