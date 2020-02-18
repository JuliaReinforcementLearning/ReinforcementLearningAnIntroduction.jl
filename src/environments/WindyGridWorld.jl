@reexport module WindyGridWorld

export WindyGridWorldEnv

using ReinforcementLearningBase

const NX = 7
const NY = 10
const Wind = [CartesianIndex(w, 0) for w in [0, 0, 0, -1, -1, -1, -2, -2, -1, 0]]
const StartPosition = CartesianIndex(4, 1)
const Goal = CartesianIndex(4, 8)
const Actions = [
    CartesianIndex(0, -1),  # left
    CartesianIndex(0, 1),   # right
    CartesianIndex(-1, 0),  # up
    CartesianIndex(1, 0),   # down
]

const LinearInds = LinearIndices((NX, NY))

Base.@kwdef mutable struct WindyGridWorldEnv <: AbstractEnv
    position::CartesianIndex{2} = StartPosition
end

RLBase.get_observation_space(env::WindyGridWorldEnv) = DiscreteSpace(length(LinearInds))
RLBase.get_action_space(env::WindyGridWorldEnv) = DiscreteSpace(length(Actions))

function (env::WindyGridWorldEnv)(a::Int)
    p = env.position + Wind[env.position[2]] + Actions[a]
    p = CartesianIndex(min(max(p[1], 1), NX), min(max(p[2], 1), NY))
    env.position = p
    nothing
end

RLBase.observe(env::WindyGridWorldEnv) =
    (
        state = LinearInds[env.position],
        terminal = env.position == Goal,
        reward = env.position == Goal ? 0.0 : -1.0,
    )

function RLBase.reset!(env::WindyGridWorldEnv)
    env.position = StartPosition
    nothing
end

end