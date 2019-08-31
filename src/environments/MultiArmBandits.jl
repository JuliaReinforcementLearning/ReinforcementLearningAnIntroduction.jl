module MultiArmBandits

export MultiArmBanditsEnv
       reset!, observe, interact!

using ReinforcementLearningEnvironments
import ReinforcementLearningEnvironments:reset!, observe, interact!

mutable struct MultiArmBanditsEnv <: AbstractEnv
    truevalues::Vector{Float64}
    truereward::Float64
    bestaction::Int
    isbest::Bool
    reward::Float64
    observation_space::DiscreteSpace
    action_space::DiscreteSpace
    function MultiArmBanditsEnv(truereward::Float64=0., k::Int=10) 
        truevalues = truereward .+ randn(k)
        new(truevalues, truereward, findmax(truevalues)[2], 0., DiscreteSpace(1), DiscreteSpace(length(truevalues)))
    end
end

## interfaces

function reset!(env::MultiArmBanditsEnv)
    env.truevalues = env.truereward .+ randn(length(env.truevalues))
    env.bestaction = findmax(env.truevalues)[2]
    env.isbest = false
    nothing
end

function interact!(env::MultiArmBanditsEnv, a::Int) 
    env.isbest = a == env.bestaction
    env.reward = randn() + env.truevalues[a]
end

observe(env::MultiArmBanditsEnv) = Observation(state=1, terminal=false, reward=env.reward)

end