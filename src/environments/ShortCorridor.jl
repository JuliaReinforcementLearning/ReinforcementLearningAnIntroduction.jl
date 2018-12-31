module ShortCorridor

using Ju

import Ju:AbstractSyncEnvironment,
          reset!, render, observe, observationspace, actionspace

export ShortCorridorEnv

mutable struct ShortCorridorEnv <: AbstractSyncEnvironment{DiscreteSpace,DiscreteSpace,1}
    position::Int
    ShortCorridorEnv() = new(1)
end

function (env::ShortCorridorEnv)(a)
    if env.position == 1 && a == 2
        env.position += 1
    elseif env.position == 2
        env.position += a == 1 ? 1 : -1
    elseif env.position == 3
        env.position += a == 1 ? -1 : 1
    end
    (observation = env.position,
     reward      = -1.,
     isdone      = env.position == 4)
end

function reset!(env::ShortCorridorEnv)
    env.position = 1
    (observation=1, isdone=false)
end

observe(env::ShortCorridorEnv) = (observation=env.position, isdone=env.position==4)
observationspace(env::ShortCorridorEnv) = DiscreteSpace(4)
actionspace(env::ShortCorridorEnv) = DiscreteSpace(2)

end