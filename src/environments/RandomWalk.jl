module RandomWalk

using Ju

import Ju:AbstractSyncEnvironment,
          reset!, render, observe, observationspace, actionspace

export RandomWalkEnv


mutable struct RandomWalkEnv <: AbstractSyncEnvironment{DiscreteSpace,DiscreteSpace,1}
    N::Int
    start::Int
    state::Int
    actions::Vector{Int}
    leftreward::Float64
    rightreward::Float64
    RandomWalkEnv(;N=7, actions=[-1, 1], start=div(N+1, 2), leftreward=-1., rightreward=1.) = new(N, start, start, actions, leftreward, rightreward)
end

function (env::RandomWalkEnv)(a::Int)
    env.state = min(max(env.state + env.actions[a], 1), env.N)
    (observation = env.state,
     reward = env.state == env.N ? env.rightreward : (env.state == 1 ? env.leftreward : 0.),
     isdone = env.state == env.N || env.state == 1)
end

function reset!(env::RandomWalkEnv)
    env.state = env.start
    (observation = env.state, isdone = false)
end

observe(env::RandomWalkEnv) = (observation = env.state, isdone = env.state == env.N || env.state == 1)
observationspace(env::RandomWalkEnv) = DiscreteSpace(env.N)
actionspace(env::RandomWalkEnv) = DiscreteSpace(length(env.actions))

end