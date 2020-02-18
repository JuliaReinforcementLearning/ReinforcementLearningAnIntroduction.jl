@reexport module RandomWalk

export RandomWalkEnv

using ReinforcementLearningBase



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

RLBase.get_observation_space(env::RandomWalkEnv) = env.observation_space
RLBase.get_action_space(env::RandomWalkEnv) = env.action_space

function (env::RandomWalkEnv)(a::Int)
    env.state = min(max(env.state + env.actions[a], 1), env.N)
    nothing
end

function RLBase.reset!(env::RandomWalkEnv)
    env.state = env.start
    nothing
end

RLBase.observe(env::RandomWalkEnv) =
    (
        state = env.state,
        terminal = env.state == env.N || env.state == 1,
        reward = env.state == env.N ? env.rightreward :
                 (env.state == 1 ? env.leftreward : 0.0),
    )

end