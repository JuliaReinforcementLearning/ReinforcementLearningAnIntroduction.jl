module WindyGridWorld

using Ju

import Ju:AbstractSyncEnvironment,
          reset!, render, observe, observationspace, actionspace

export WindyGridWorldEnv

const NX = 7
const NY = 10
const Wind = [CartesianIndex(w, 0) for w in [0, 0, 0, -1, -1, -1, -2, -2, -1, 0]]
const StartPosition = CartesianIndex(4, 1)
const Goal = CartesianIndex(4, 8)
const Actions = [CartesianIndex(0, -1),  # left
                 CartesianIndex(0, 1),   # right
                 CartesianIndex(-1, 0),  # up
                 CartesianIndex(1, 0),   # down
                ]

const LinearInds = LinearIndices((NX, NY))

mutable struct WindyGridWorldEnv <: AbstractSyncEnvironment{DiscreteSpace,DiscreteSpace,1}
    position::CartesianIndex{2}
    WindyGridWorldEnv() = new(StartPosition)
end

function (env::WindyGridWorldEnv)(a::Int)
    p = env.position + Wind[env.position[2]] + Actions[a]
    p = CartesianIndex(min(max(p[1], 1), NX), min(max(p[2], 1), NY))
    env.position = p
    (observation = LinearInds[p],
     reward      = p == Goal ? 0. : -1.,
     isdone      = p == Goal)
end

observe(env::WindyGridWorldEnv) = (observation = LinearInds[env.position], isdone = env.position == Goal)

function reset!(env::WindyGridWorldEnv)
    env.position = StartPosition
    (observation = LinearInds[StartPosition],
     isdone      = false)
end

observationspace(env::WindyGridWorldEnv) = DiscreteSpace(length(LinearInds))
actionspace(env::WindyGridWorldEnv) = DiscreteSpace(length(Actions))

end