module WindyGridWorld

export WindyGridWorldEnv, reset!, observe, interact!, get_legal_actions

using ReinforcementLearningEnvironments
import ReinforcementLearningEnvironments: reset!, observe, interact!, get_legal_actions

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

mutable struct WindyGridWorldEnv <: AbstractEnv
    position::CartesianIndex{2}
    observation_space::DiscreteSpace
    action_space::DiscreteSpace
    WindyGridWorldEnv() =
        new(
            StartPosition,
            DiscreteSpace(length(LinearInds)),
            DiscreteSpace(length(Actions)),
        )
end

function interact!(env::WindyGridWorldEnv, a::Int)
    p = env.position + Wind[env.position[2]] + Actions[a]
    p = CartesianIndex(min(max(p[1], 1), NX), min(max(p[2], 1), NY))
    env.position = p
    nothing
end

observe(env::WindyGridWorldEnv) =
    Observation(
        state = LinearInds[env.position],
        terminal = env.position == Goal,
        reward = env.position == Goal ? 0.0 : -1.0,
    )

function reset!(env::WindyGridWorldEnv)
    env.position = StartPosition
    nothing
end

end