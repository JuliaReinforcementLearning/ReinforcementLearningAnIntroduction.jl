module MountainCar

export MountainCarEnv

using ReinforcementLearningCore


const ACTIONS = [-1, 0, 1]
const POSITION_MIN = -1.2
const POSITION_MAX = 0.5
const VELOCITY_MIN = -0.07
const VELOCITY_MAX = 0.07

Base.@kwdef mutable struct MountainCarEnv <: AbstractEnv
    p::Float64 = -0.6 + rand() / 5,
    v::Float64 = 0.0
end

RLBase.get_observation_space(env::MountainCarEnv) = MultiContinuousSpace(
    [POSITION_MIN, VELOCITY_MIN],
    [POSITION_MAX, VELOCITY_MAX]
)

RLBase.get_action_space(env::MountainCarEnv) = DiscreteSpace(length(ACTIONS))

function (env::MountainCarEnv)(a)
    action = ACTIONS[a]
    v′ = min(
        max(VELOCITY_MIN, env.v + 0.001 * action - 0.0025 * cos(3 * env.p)),
        VELOCITY_MAX,
    )
    p′ = min(max(POSITION_MIN, env.p + v′), POSITION_MAX)
    if p′ == POSITION_MIN
        v′ = 0.0
    end
    env.p = p′
    env.v = v′
    nothing
end

RLBase.observe(env::MountainCarEnv) =
    (
        reward = env.p == POSITION_MAX ? 0.0 : -1.0,
        terminal = env.p == POSITION_MAX,
        state = [env.p, env.v],
    )

function RLBase.reset!(env::MountainCarEnv)
    env.p = -0.6 + rand() / 5
    env.v = 0.0
    nothing
end

end