@reexport module Maze

export MazeEnv

using ReinforcementLearningBase

import Base: *

const Actions = [
    CartesianIndex(0, -1),  # left
    CartesianIndex(0, 1),   # right
    CartesianIndex(-1, 0),  # up
    CartesianIndex(1, 0),   # down
]

mutable struct MazeEnv <: AbstractEnv
    walls::Set{CartesianIndex{2}}
    position::CartesianIndex{2}
    start::CartesianIndex{2}
    goal::CartesianIndex{2}
    NX::Int
    NY::Int
    observation_space::DiscreteSpace
    action_space::DiscreteSpace
    MazeEnv(w, p, s, g, NX, NY) =
        new(w, p, s, g, NX, NY, DiscreteSpace(NX * NY), DiscreteSpace(length(Actions)))
end

RLBase.get_observation_space(env::MazeEnv) = env.observation_space
RLBase.get_action_space(env::MazeEnv) = env.action_space

function MazeEnv()
    walls = Set([
        [CartesianIndex(i, 3) for i = 2:4]
        CartesianIndex(5, 6)
        [CartesianIndex(j, 8) for j = 1:3]
    ])
    start = CartesianIndex(3, 1)
    goal = CartesianIndex(1, 9)
    MazeEnv(walls, start, start, goal, 6, 9)
end

function extend(p::CartesianIndex{2}, n::Int)
    x, y = Tuple(p)
    [CartesianIndex(n * (x - 1) + i, n * (y - 1) + j) for i = 1:n for j = 1:n]
end

function remap(p::CartesianIndex{2}, n::Int)
    x, y = Tuple(p)
    CartesianIndex((x - 1) * n + 1, (y - 1) * n + 1)
end

function *(env::MazeEnv, n::Int)
    walls = Set{CartesianIndex{2}}(ww for w in env.walls for ww in extend(w, n))
    start, position, goal = remap(env.start, n), remap(env.position, n), remap(env.goal, n)
    NX, NY = env.NX * n, env.NY * n
    MazeEnv(walls, position, start, goal, NX, NY)
end

function (env::MazeEnv)(a::Int)
    p = env.position + Actions[a]
    if p == env.goal
        env.position = env.goal
    elseif !(p ∈ env.walls)
        env.position = CartesianIndex(min(max(p[1], 1), env.NX), min(max(p[2], 1), env.NY))
    end
    nothing
end

RLBase.observe(env::MazeEnv) =
    (
        reward = Float64(env.position == env.goal),
        terminal = env.position == env.goal,
        state = (env.position[2] - 1) * env.NX + env.position[1],
    )

function RLBase.reset!(env::MazeEnv)
    env.position = env.start
    nothing
end

end