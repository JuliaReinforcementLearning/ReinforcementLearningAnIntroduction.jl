module ShortCorridor

export ShortCorridorEnv,
       reset!, observe, interact!

using ReinforcementLearningEnvironments
import ReinforcementLearningEnvironments:reset!, observe, interact!

mutable struct ShortCorridorEnv <: AbstractEnv
    position::Int
    observation_space::DiscreteSpace
    action_space::DiscreteSpace
    ShortCorridorEnv() = new(1, DiscreteSpace(4), DiscreteSpace(2))
end

function interact!(env::ShortCorridorEnv, a)
    if env.position == 1 && a == 2
        env.position += 1
    elseif env.position == 2
        env.position += a == 1 ? 1 : -1
    elseif env.position == 3
        env.position += a == 1 ? -1 : 1
    end
    nothing
end

function reset!(env::ShortCorridorEnv)
    env.position = 1
    nothing
end

observe(env::ShortCorridorEnv) = Observation(
    state = env.position,
    terminal = env.position==4,
    reward = env.position == 4 ? 0. : -1.
)

end