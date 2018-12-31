module MaximizationBias

using Ju
import Ju:AbstractSyncEnvironment,
          reset!, render, observe, observationspace, actionspace

export MaximizationBiasEnv

"""
states:
1:A
2:B
3:terminal

actions:
1: left
2: right
"""
mutable struct MaximizationBiasEnv <: AbstractSyncEnvironment{DiscreteSpace,DiscreteSpace,1}
    position::Int
    MaximizationBiasEnv() = new(1)
end

function (env::MaximizationBiasEnv)(a::Int)
    if env.position == 1
        if a == 1
            env.position = 2
            (observation=env.position, reward=0., isdone=false)
        else
            env.position = 3
            (observation=env.position, reward=0., isdone=true)
        end
    elseif env.position == 2
        env.position = 3
        (observation=3, reward=randn()-0.1, isdone=true)
    end
end

function reset!(env::MaximizationBiasEnv)
    env.position = 1
    (observation=1, isdone=false)
end

observe(env::MaximizationBiasEnv) = (observation=env.position, isdone=env.position == 3)
observationspace(env::MaximizationBiasEnv) = DiscreteSpace(3)
actionspace(env::MaximizationBiasEnv) = DiscreteSpace(10)

end