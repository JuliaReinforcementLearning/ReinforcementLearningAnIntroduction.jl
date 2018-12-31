module BranchMDP

using Ju

import Ju:AbstractSyncEnvironment,
          reset!, render, observe, observationspace, actionspace

export BranchMDPEnv

mutable struct BranchMDPEnv <: AbstractSyncEnvironment{DiscreteSpace, DiscreteSpace, 1}
    transition::Array{Int, 3}
    rewards::Array{Float64, 3}
    current::Int
    termination_prob::Float64
    BranchMDPEnv(ns::Int, na::Int, b::Int, termination_prob::Float64=0.1) = new(
        rand(1:ns, ns, na, b),
        randn(ns, na, b),
        1,
        termination_prob,
        false)
end

function (env::BranchMDPEnv)(a::Int)
    if rand() < env.termination_prob
        (observation=size(env.transition, 1) + 1, reward=0., isdone=true)
    else
        b = rand(1:size(env.transition, 3))
        s′ = env.transition[env.current, a, b]
        r = env.rewards[env.current, a, b]
        env.current = s′
        (observation=s′, reward=r, isdone=false)
    end
end

observe(env::BranchMDPEnv) = (observation=env.current, isdone=env.current == size(env.transition, 1)+1)

function reset!(env::BranchMDPEnv, s::Int=1)
    env.current = s
    (observation=s, isdone=false)
end

observationspace(env::BranchMDPEnv) = DiscreteSpace(size(env.transition, 1)+1)
actionspace(env::BranchMDPEnv) = DiscreteSpace(size(env.transition, 2))

end