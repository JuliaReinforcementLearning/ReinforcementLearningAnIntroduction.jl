module MultiArmBandits

using Ju
using StatPlots
gr()

import Ju:AbstractSyncEnvironment,
          reset!, render, observe, observationspace, actionspace

export MultiArmBanditsEnv

mutable struct MultiArmBanditsEnv <: AbstractSyncEnvironment{DiscreteSpace, DiscreteSpace, 1}
    truevalues::Vector{Float64}
    truereward::Float64
    bestaction::Int
    isbest::Bool
    function MultiArmBanditsEnv(truereward::Float64=0., k::Int=10) 
        truevalues = truereward .+ randn(k)
        new(truevalues, truereward, findmax(truevalues)[2])
    end
end

## interfaces

function reset!(env::MultiArmBanditsEnv)
    env.truevalues = env.truereward .+ randn(length(env.truevalues))
    env.bestaction = findmax(env.truevalues)[2]
    env.isbest = false
    (observation=1, isdone=false)
end

function (env::MultiArmBanditsEnv)(a::Int) 
    env.isbest = a == env.bestaction
    (observation=1,
     reward=randn() + env.truevalues[a],
     isdone=false)
end

"`violin` is broken. https://github.com/JuliaPlots/StatPlots.jl/issues/198"
render(env::MultiArmBanditsEnv) = violin([randn(100) .+ x for x in env.truevalues], leg=false)
observe(env::MultiArmBanditsEnv) = (observation=1, isdone=false)
observationspace(env::MultiArmBanditsEnv) = DiscreteSpace(1)
actionspace(env::MultiArmBanditsEnv) = DiscreteSpace(length(env.truevalues))

end