@reexport module AccessControl

export AccessControlEnv

using ReinforcementLearningBase


using Distributions


const N_SERVERS = 10
const PRIORITIES = [1.0, 2.0, 4.0, 8.0]
const FREE_PROB = 0.06
const ACTIONS = (:accept, :reject)
const CUSTOMERS = 1:length(PRIORITIES)

const TRANSFORMER = LinearIndices((0:N_SERVERS, CUSTOMERS))

Base.@kwdef mutable struct AccessControlEnv <: AbstractEnv
    n_servers::Int = 10
    n_free_servers::Int = 0
    customer::Int = rand(CUSTOMERS)
    reward::Float64 = 0.0
end

RLBase.get_observation_space(env::AccessControlEnv) = DiscreteSpace(length(TRANSFORMER))
RLBase.get_action_space(env::AccessControlEnv) = DiscreteSpace(2)

function (env::AccessControlEnv)(a)
    action, reward = ACTIONS[a], 0.0
    if env.n_free_servers > 0 && action == :accept
        env.n_free_servers -= 1
        reward = PRIORITIES[env.customer]
    end

    env.n_free_servers += rand(Binomial(env.n_servers - env.n_free_servers, FREE_PROB))
    env.customer = rand(CUSTOMERS)
    env.reward = reward

    nothing
end

RLBase.observe(env::AccessControlEnv) =
    (
        reward = env.reward,
        terminal = false,
        state = TRANSFORMER[CartesianIndex(env.n_free_servers + 1, env.customer)],
    )

function RLBase.reset!(env::AccessControlEnv)
    env.n_free_servers = env.n_servers
    env.customer = rand(CUSTOMERS)
    env.reward = 0.0
    nothing
end

end