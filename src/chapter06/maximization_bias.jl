using Ju
using ..MaximizationBias
using StatsBase:mean
using Plots
gr()

figpath(f) = "docs/src/assets/figures/figure_$f.png"

function count_left_actions_from_A()
    counts_per_episode = []
    counts = 0
    function update(env, agent)
        if agent.buffer.state[end-1] == 1 && agent.buffer.action[end-1] == 1
            counts += 1
        end
        if agent.buffer.isdone[end]
            push!(counts_per_episode, counts)
            counts = 0
        end
    end
    function update()
        counts_per_episode
    end
end

function gen_env_DQagent()
    env = MaximizationBiasEnv()
    n_states = length(observationspace(env))
    n_actions = length(actionspace(env))
    p1 = EpsilonGreedyPolicy(rand(1:n_actions, n_states), n_actions, 0.1)
    Q1 = OffPolicyTDLearner(TabularQ(n_states, n_actions), p1, p1, 1.0, 0.1, 0, :QLearning)
    p2 = EpsilonGreedyPolicy(rand(1:n_actions, n_states), n_actions, 0.1)
    Q2 = OffPolicyTDLearner(TabularQ(n_states, n_actions), p2, p2, 1.0, 0.1, 0, :QLearning)
    agent = Agent(DoubleLearner(Q1, Q2, EpsilonGreedySelector(0.1)), EpisodeSARDBuffer())
    env, agent
end

function gen_env_Qagent()
    env = MaximizationBiasEnv()
    n_states = length(observationspace(env))
    n_actions = length(actionspace(env))
    p1 = EpsilonGreedyPolicy(rand(1:n_actions, n_states), n_actions, 0.1)
    Q1 = OffPolicyTDLearner(TabularQ(n_states, n_actions), p1, p1, 1.0, 0.1, 0, :QLearning)
    agent = Agent(Q1, EpisodeSARDBuffer())
    env, agent
end

function fig_6_5()
    function run_once(env, agent)
        cbs = (stop_at_episode(300), count_left_actions_from_A())
        train!(env, agent; callbacks=cbs)
        cbs[2]()
    end
    p = plot(legend=:topright, dpi=200)
    plot!(p, mean(run_once(gen_env_DQagent()...) for _ in 1:10000), label="Double-Q")
    plot!(p, mean(run_once(gen_env_Qagent()...) for _ in 1:10000), label="Q")
    savefig(p, figpath("6_5"))
    p
end