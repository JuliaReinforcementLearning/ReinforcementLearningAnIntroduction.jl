module MountainCar

using Ju

import Ju:AbstractSyncEnvironment,
          reset!, render, observe, observationspace, actionspace

export MountainCarEnv

const ACTIONS = [-1, 0, 1]
const POSITION_MIN = -1.2
const POSITION_MAX = 0.5
const VELOCITY_MIN = -0.07
const VELOCITY_MAX = 0.07

mutable struct MountainCarEnv <: AbstractSyncEnvironment{MultiContinuousSpace, DiscreteSpace, 1}
    p::Float64
    v::Float64
    MountainCarEnv() = new(-0.6 + rand() / 5, 0.)
end

function (env::MountainCarEnv)(a)
    action = ACTIONS[a]
    v′ = min(max(VELOCITY_MIN, env.v + 0.001 * action - 0.0025 * cos(3 * env.p)), VELOCITY_MAX)
    p′ = min(max(POSITION_MIN, env.p + v′), POSITION_MAX)
    if p′ == POSITION_MIN
        v′ = 0.
    end
    env.p = p′
    env.v = v′
    (observation = [env.p, env.v],
     reward      = env.p == POSITION_MAX ? 0. : -1.,
     isdone      = env.p == POSITION_MAX)
end

observe(env::MountainCarEnv) = (observation = [env.p, env.v], isdone = env.p == POSITION_MAX)

function reset!(env::MountainCarEnv)
    env.p = -0.6 + rand() / 5
    env.v = 0.
    (observation = [env.p, env.v], isdone = false)
end

observationspace(env::MountainCarEnv) = MultiContinuousSpace([POSITION_MIN, VELOCITY_MIN], [POSITION_MAX, VELOCITY_MAX])
actionspace(env::MountainCarEnv) = DiscreteSpace(length(ACTIONS))

end