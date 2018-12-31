module LeftRight

using Ju
using StatsBase:sample, Weights

import Ju:AbstractSyncEnvironment,
          reset!, render, observe, observationspace, actionspace

export LeftRightEnv

mutable struct LeftRightEnv <: AbstractSyncEnvironment{DiscreteSpace,DiscreteSpace,1}
    transitions::Array{Float64,3}
    current_state::Int
end

function LeftRightEnv()
    t = zeros(2, 2, 2)
    t[1, :, :] = [0.9 0.1; 0. 1.]
    t[2, :, :] = [0. 1.; 0. 1.]
    LeftRightEnv(t, rand(1:2))
end

function(env::LeftRightEnv)(a::Int)
    s′ = sample(Weights(@view(env.transitions[env.current_state, a, :]), 1.))
    env.current_state = s′
    (observation = s′,
     reward      = Float64(s′ == 2),
     isdone      = s′ == 2)
end

function reset!(env::LeftRightEnv) 
    env.current_state = 1
    (observation = 1,
     isdone = false)
end

observe(env::LeftRightEnv) = (observation = env.current_state, isdone = env.current_state == 2)
observationspace(env::LeftRightEnv) = DiscreteSpace(2)
actionspace(env::LeftRightEnv) = DiscreteSpace(2)

end