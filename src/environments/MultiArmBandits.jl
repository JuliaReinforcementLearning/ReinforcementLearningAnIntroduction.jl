module MultiArmBandits

export MultiArmBanditsEnv,
       reset!, observe, interact!

using ReinforcementLearningEnvironments
import ReinforcementLearningEnvironments:reset!, observe, interact!

mutable struct MultiArmBanditsEnv <: AbstractEnv
    truevalues::Vector{Float64}
    truereward::Float64
    bestaction::Int
    isbest::Bool
    reward::Float64
    isdone::Bool
    observation_space::DiscreteSpace
    action_space::DiscreteSpace
    function MultiArmBanditsEnv(;truereward::Float64=0., k::Int=10) 
        truevalues = truereward .+ randn(k)
        new(truevalues, truereward, findmax(truevalues)[2], false, 0., false, DiscreteSpace(1), DiscreteSpace(length(truevalues)))
    end
end

## interfaces

function reset!(env::MultiArmBanditsEnv)
    env.isbest = false
    nothing
end

function interact!(env::MultiArmBanditsEnv, a::Int) 
    env.isbest = a == env.bestaction
    env.reward = randn() + env.truevalues[a]
    env.isdone = true
    nothing
end

observe(env::MultiArmBanditsEnv) = Observation(state=1, terminal=env.isdone, reward=env.reward)

end