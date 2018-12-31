module BairdCounter

using Ju

import Ju:AbstractSyncEnvironment,
          reset!, render, observe, observationspace, actionspace

export BairdCounterEnv

const ACTIONS = (:dashed, :solid)

mutable struct BairdCounterEnv <: AbstractSyncEnvironment{DiscreteSpace, DiscreteSpace, 1}
    current::Int
    BairdCounterEnv() = new(rand(1:7))
end

function (env::BairdCounterEnv)(a)
    if ACTIONS[a] == :dashed
        env.current = rand(1:6)
    else
        env.current = 7
    end
    (observation = env.current,
     reward      = 0.,
     isdone      = false)
end

observe(env::BairdCounterEnv) = (observation=env.current, isdone=false)

function reset!(env::BairdCounterEnv)
    env.current = rand(1:6)
    (observation=env.current, isdone=false)
end

observationspace(env::BairdCounterEnv) = DiscreteSpace(7)
actionspace(env::BairdCounterEnv) = DiscreteSpace(length(ACTIONS))

end