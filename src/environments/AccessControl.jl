module AccessControl

export AccessControlEnv,
       reset!, observe, interact!

using ReinforcementLearningEnvironments
import ReinforcementLearningEnvironments:reset!, observe, interact!

using Distributions


const N_SERVERS = 10
const PRIORITIES = [1.,2.,4.,8.]
const FREE_PROB = 0.06
const ACTIONS = (:accept, :reject)
const CUSTOMERS = 1:length(PRIORITIES)

const TRANSFORMER = LinearIndices((0:N_SERVERS, CUSTOMERS))

mutable struct AccessControlEnv <: AbstractEnv
    n_servers::Int
    n_free_servers::Int
    customer::Int
    reward::Float64
    observation_space::DiscreteSpace
    action_space::DiscreteSpace
    AccessControlEnv() = new(10, 0, rand(CUSTOMERS), 0., DiscreteSpace(length(TRANSFORMER)), DiscreteSpace(2))
end

function interact!(env::AccessControlEnv, a)
    action, reward = ACTIONS[a], 0.
    if env.n_free_servers > 0 && action == :accept
        env.n_free_servers -= 1
        reward = PRIORITIES[env.customer]
    end

    env.n_free_servers += rand(Binomial(env.n_servers - env.n_free_servers, FREE_PROB))
    env.customer = rand(CUSTOMERS)
    env.reward = reward

    nothing
end

observe(env::AccessControlEnv) = Observation(
    reward = env.reward,
    terminal = false,
    state = TRANSFORMER[CartesianIndex(env.n_free_servers+1, env.customer)]
)

function reset!(env::AccessControlEnv)
    env.n_free_servers = env.n_servers
    env.customer = rand(CUSTOMERS)
    env.reward = 0.
    nothing
end

end