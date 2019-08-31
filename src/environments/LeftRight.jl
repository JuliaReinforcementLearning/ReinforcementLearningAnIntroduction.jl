module LeftRight

export LeftRightEnv,
       reset!, observe, interact!

using ReinforcementLearningEnvironments
import ReinforcementLearningEnvironments:reset!, observe, interact!

mutable struct LeftRightEnv <: AbstractEnv
    transitions::Array{Float64,3}
    current_state::Int
    observation_space::DiscreteSpace
    actionspace::DiscreteSpace
    LeftRightEnv(transitions, current_state) = new(transitions, current_state, DiscreteSpace(2), DiscreteSpace(2))
end

function LeftRightEnv()
    t = zeros(2, 2, 2)
    t[1, :, :] = [0.9 0.1; 0. 1.]
    t[2, :, :] = [0. 1.; 0. 1.]
    LeftRightEnv(t, rand(1:2))
end

function interact!(env::LeftRightEnv, a::Int)
    env.current_state = sample(Weights(@view(env.transitions[env.current_state, a, :]), 1.))
    nothing
end

function reset!(env::LeftRightEnv) 
    env.current_state = 1
    nothing
end

observe(env::LeftRightEnv) = Observation(
    reward = Float64(env.current_state == 2),
    terminal = env.current_state == 2,
    state = env.current_state
)

end