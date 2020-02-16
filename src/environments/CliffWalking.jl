@reexport module CliffWalking

export CliffWalkingEnv

using ReinforcementLearningBase


const NX = 4
const NY = 12
const Start = CartesianIndex(4, 1)
const Goal = CartesianIndex(4, 12)
const Actions = [
    CartesianIndex(0, -1),  # left
    CartesianIndex(0, 1),   # right
    CartesianIndex(-1, 0),  # up
    CartesianIndex(1, 0),   # down
]

const LinearInds = LinearIndices((NX, NY))

function iscliff(p::CartesianIndex{2})
    x, y = Tuple(p)
    x == 4 && y > 1 && y < NY
end

Base.@kwdef mutable struct CliffWalkingEnv <: AbstractEnv
    position::CartesianIndex{2} = Start
end

RLBase.get_observation_space(env::CliffWalkingEnv) = DiscreteSpace(length(LinearInds))
RLBase.get_action_space(env::CliffWalkingEnv) = DiscreteSpace(length(Actions))

function (env::CliffWalkingEnv)(a::Int)
    x, y = Tuple(env.position + Actions[a])
    env.position = CartesianIndex(min(max(x, 1), NX), min(max(y, 1), NY))
    nothing
end

RLBase.observe(env::CliffWalkingEnv) =
    (
        reward = env.position == Goal ? 0.0 : (iscliff(env.position) ? -100.0 : -1.0),
        terminal = env.position == Goal || iscliff(env.position),
        state = LinearInds[env.position],
    )

function RLBase.reset!(env::CliffWalkingEnv)
    env.position = Start
    nothing
end

end