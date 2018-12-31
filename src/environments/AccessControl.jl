module AccessControl

using Ju
using Distributions

import Ju:AbstractSyncEnvironment,
          reset!, render, observe, observationspace, actionspace

export AccessControlEnv

const N_SERVERS = 10
const PRIORITIES = [1.,2.,4.,8.]
const FREE_PROB = 0.06
const ACTIONS = (:accept, :reject)
const CUSTOMERS = 1:length(PRIORITIES)

const TRANSFORMER = LinearIndices((0:N_SERVERS, CUSTOMERS))

mutable struct AccessControlEnv <: AbstractSyncEnvironment{DiscreteSpace, DiscreteSpace, 1}
    n_servers::Int
    n_free_servers::Int
    customer::Int
    AccessControlEnv() = new(10, 0, rand(CUSTOMERS))
end

function (env::AccessControlEnv)(a)
    action, reward = ACTIONS[a], 0.
    if env.n_free_servers > 0 && action == :accept
        env.n_free_servers -= 1
        reward = PRIORITIES[env.customer]
    end

    env.n_free_servers += rand(Binomial(env.n_servers - env.n_free_servers, FREE_PROB))
    env.customer = rand(CUSTOMERS)

    (observation = TRANSFORMER[CartesianIndex(env.n_free_servers+1, env.customer)],
     reward      = reward,
     isdone      = false)
end

observe(env::AccessControlEnv) = (observation=TRANSFORMER[CartesianIndex(env.n_free_servers+1, env.customer)], isdone=false)

function reset!(env::AccessControlEnv)
    env.n_free_servers = env.n_servers
    env.customer = rand(CUSTOMERS)
    (observation=TRANSFORMER[CartesianIndex(env.n_free_servers+1, env.customer)], isdone=false)
end

observationspace(env::AccessControlEnv) = DiscreteSpace(length(TRANSFORMER))
actionspace(env::AccessControlEnv) = DiscreteSpace(2)

end