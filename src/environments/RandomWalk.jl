module RandomWalk

export RandomWalkEnv, reset!, observe, interact!

using ReinforcementLearningEnvironments
import ReinforcementLearningEnvironments: reset!, observe, interact!


mutable struct RandomWalkEnv <: AbstractEnv
    N::Int
    start::Int
    state::Int
    actions::Vector{Int}
    leftreward::Float64
    rightreward::Float64
    observation_space::DiscreteSpace
    action_space::DiscreteSpace

    RandomWalkEnv(
        ;
        N = 7,
        actions = [-1, 1],
        start = div(N + 1, 2),
        leftreward = -1.0,
        rightreward = 1.0,
    ) =
        new(
            N,
            start,
            start,
            actions,
            leftreward,
            rightreward,
            DiscreteSpace(N),
            DiscreteSpace(length(actions)),
        )
end

function interact!(env::RandomWalkEnv, a::Int)
    env.state = min(max(env.state + env.actions[a], 1), env.N)
    nothing
end

function reset!(env::RandomWalkEnv)
    env.state = env.start
    nothing
end

observe(env::RandomWalkEnv) =
    Observation(
        state = env.state,
        terminal = env.state == env.N || env.state == 1,
        reward = env.state == env.N ? env.rightreward :
                 (env.state == 1 ? env.leftreward : 0.0),
    )

end