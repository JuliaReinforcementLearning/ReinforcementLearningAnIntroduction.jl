module MountainCar

export MountainCarEnv,
       reset!, observe, interact!

using ReinforcementLearningEnvironments
import ReinforcementLearningEnvironments:reset!, observe, interact!

const ACTIONS = [-1, 0, 1]
const POSITION_MIN = -1.2
const POSITION_MAX = 0.5
const VELOCITY_MIN = -0.07
const VELOCITY_MAX = 0.07

mutable struct MountainCarEnv <: AbstractEnv
    p::Float64
    v::Float64
    observation_space::MultiContinuousSpace
    action_space::DiscreteSpace
    MountainCarEnv() = new(-0.6 + rand() / 5, 0., MultiContinuousSpace([POSITION_MIN, VELOCITY_MIN], [POSITION_MAX, VELOCITY_MAX]), DiscreteSpace(length(ACTIONS)))
end

function interact!(env::MountainCarEnv, a)
    action = ACTIONS[a]
    v′ = min(max(VELOCITY_MIN, env.v + 0.001 * action - 0.0025 * cos(3 * env.p)), VELOCITY_MAX)
    p′ = min(max(POSITION_MIN, env.p + v′), POSITION_MAX)
    if p′ == POSITION_MIN
        v′ = 0.
    end
    env.p = p′
    env.v = v′
    nothing
end

observe(env::MountainCarEnv) = Observation(
    reward = env.p == POSITION_MAX ? 0. : -1.,
    terminal = env.p == POSITION_MAX,
    state = [env.p, env.v]
)

function reset!(env::MountainCarEnv)
    env.p = -0.6 + rand() / 5
    env.v = 0.
    nothing
end

end