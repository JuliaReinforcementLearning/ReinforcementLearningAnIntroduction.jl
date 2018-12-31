module CliffWalking

using Ju

import Ju:AbstractSyncEnvironment,
          reset!, render, observe, observationspace, actionspace

export CliffWalkingEnv

const NX = 4
const NY = 12
const Start = CartesianIndex(4, 1)
const Goal = CartesianIndex(4, 12)
const Actions = [CartesianIndex(0, -1),  # left
                 CartesianIndex(0, 1),   # right
                 CartesianIndex(-1, 0),  # up
                 CartesianIndex(1, 0),   # down
                ]

const LinearInds = LinearIndices((NX, NY))

function iscliff(p::CartesianIndex{2})
    x, y = Tuple(p)
    x == 4 && y > 1 && y < NY
end

mutable struct CliffWalkingEnv <: AbstractSyncEnvironment{DiscreteSpace,DiscreteSpace,1}
    position::CartesianIndex{2}
    CliffWalkingEnv() = new(Start)
end

function (env::CliffWalkingEnv)(a::Int)
    x, y = Tuple(env.position + Actions[a])
    p = CartesianIndex(min(max(x, 1), NX), min(max(y, 1), NY))
    env.position = p
    (observation = LinearInds[p],
     reward      = p == Goal ? 0. : (iscliff(p) ? -100. : -1.),
     isdone      = p == Goal || iscliff(p))
end

observe(env::CliffWalkingEnv) = (observation=LinearInds[env.position], isdone = env.position == Goal || iscliff(env.position))

function reset!(env::CliffWalkingEnv)
    env.position = Start
    (observation = LinearInds[Start],
     isdone      = false)
end

observationspace(env::CliffWalkingEnv) = DiscreteSpace(length(LinearInds))
actionspace(env::CliffWalkingEnv) = DiscreteSpace(length(Actions))

end