module Maze

using Ju

import Ju:AbstractSyncEnvironment,
          reset!, render, observe, observationspace, actionspace
import Base:*

export MazeEnv

const Actions = [CartesianIndex(0, -1),  # left
                 CartesianIndex(0, 1),   # right
                 CartesianIndex(-1, 0),  # up
                 CartesianIndex(1, 0),   # down
                ]

mutable struct MazeEnv <: AbstractSyncEnvironment{DiscreteSpace, DiscreteSpace, 1}
    walls::Set{CartesianIndex{2}}
    position::CartesianIndex{2}
    start::CartesianIndex{2}
    goal::CartesianIndex{2}
    NX::Int
    NY::Int
end

function extend(p::CartesianIndex{2}, n::Int)
    x, y = Tuple(p)
    [CartesianIndex(n*(x-1)+i, n*(y-1)+j) for i in 1:n for j in 1:n]
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

function MazeEnv()
    walls = Set([[CartesianIndex(i,3) for i in 2:4]; CartesianIndex(5, 6); [CartesianIndex(j, 8) for j in 1:3]])
    start = CartesianIndex(3, 1)
    goal = CartesianIndex(1, 9)
    MazeEnv(walls, start, start, goal, 6, 9)
end

function (env::MazeEnv)(a::Int)
    p, reward, isdone = env.position + Actions[a], 0., false
    if p == env.goal
        reward = 1.
        isdone = true
        env.position = env.goal
    elseif !(p ∈ env.walls)
        env.position = CartesianIndex(min(max(p[1], 1), env.NX), min(max(p[2], 1), env.NY))
    end
    (observation = (env.position[2]-1)*env.NX + env.position[1],
     reward      = reward,
     isdone      = isdone)
end

observe(env::MazeEnv) = (observation = (env.position[2]-1)*env.NX + env.position[1], isdone = env.position == env.goal)
function reset!(env::MazeEnv)
    env.position = env.start
    (observation = (env.position[2]-1)*env.NX + env.position[1], isdone = env.position == env.goal)
end

observationspace(env::MazeEnv) = DiscreteSpace(env.NX * env.NY)
actionspace(env::MazeEnv) = DiscreteSpace(length(Actions))

end