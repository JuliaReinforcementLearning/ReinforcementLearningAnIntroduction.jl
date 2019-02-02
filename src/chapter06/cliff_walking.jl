using Ju
using StatsBase:mean
using ..CliffWalking
using Plots
gr()



function rewards_of_each_episode()
    rewards = []
    sum_of_reward = 0
    function update(env, agent)
        sum_of_reward += agent.buffer.reward[end]
        if agent.buffer.isdone[end]
            push!(rewards, sum_of_reward)
            sum_of_reward = 0
        end
    end
    function update()
        rewards
    end
end

function gen_env_ExpectedSARSAagent(α=0.5) 
    env = CliffWalkingEnv()
    n_states = length(observationspace(env))
    n_actions = length(actionspace(env))
    policy = EpsilonGreedyPolicy(rand(1:n_actions, n_states), n_actions, 0.1)
    agent = Agent(TDLearner(TabularQ(n_states, n_actions), policy, 1.0, α, 0, :ExpectedSARSA),
                  EpisodeSARDBuffer())
    env, agent
end

function gen_env_SARSAagent(α=0.5) 
    env = CliffWalkingEnv()
    n_states = length(observationspace(env))
    n_actions = length(actionspace(env))
    policy = EpsilonGreedyPolicy(rand(1:n_actions, n_states), n_actions, 0.1)
    agent = Agent(TDLearner(TabularQ(n_states, n_actions), policy, 1.0, α),
                  EpisodeSARDBuffer())
    env, agent
end

function gen_env_Qagent(α=0.5) 
    env = CliffWalkingEnv()
    n_states = length(observationspace(env))
    n_actions = length(actionspace(env))
    policy = EpsilonGreedyPolicy(rand(1:n_actions, n_states), n_actions, 0.1)
    agent = Agent(OffPolicyTDLearner(TabularQ(n_states, n_actions), policy, policy, 1.0, α, 0, :QLearning),
                  EpisodeSARDBuffer())
    env, agent
end

function fig_6_3_a(fig_dir=".")
    function rewards(env, agent)
        callbacks=(stop_at_episode(500, false), rewards_of_each_episode())
        train!(env, agent; callbacks=callbacks)
        callbacks[2]()
    end

    p = plot(legend=:bottomright, dpi=200)
    plot!(p, mean(rewards(gen_env_Qagent()...) for _ in 1:100), label="QLearning")
    plot!(p, mean(rewards(gen_env_SARSAagent()...) for _ in 1:100), label="SARSA")
    savefig(p, joinpath(fig_dir, "figure_6_3_a.png"))
    p
end

function fig_6_3_b(fig_dir=".")
    A = 0.1:0.05:0.95
    function avg_reward_per_episode(n_episodes, env, agent)
        callbacks=(stop_at_episode(n_episodes, false), rewards_of_each_episode())
        train!(env, agent; callbacks=callbacks)
        mean(callbacks[2]())
    end

    p = plot(legend=:bottomright, dpi=200)

    plot!(p, A, [mean(avg_reward_per_episode(100, gen_env_Qagent(α)...) for _ in 1:100) for α in A], label="Interim Q")
    plot!(p, A, [mean(avg_reward_per_episode(100, gen_env_SARSAagent(α)...) for _ in 1:100) for α in A], label="Interim SARSA")
    plot!(p, A, [mean(avg_reward_per_episode(100, gen_env_ExpectedSARSAagent(α)...) for _ in 1:100) for α in A], label="Interim ExpectedSARSA")

    plot!(p, A, [mean(avg_reward_per_episode(1000, gen_env_Qagent(α)...) for _ in 1:10) for α in A], label="Asymptotic interim Q")
    plot!(p, A, [mean(avg_reward_per_episode(1000, gen_env_SARSAagent(α)...) for _ in 1:10) for α in A], label="Asymptotic SARSA")
    plot!(p, A, [mean(avg_reward_per_episode(1000, gen_env_ExpectedSARSAagent(α)...) for _ in 1:10) for α in A], label="Asymptotic ExpectedSARSA")
    savefig(p, joinpath(fig_dir, "figure_6_3_b.png"))
    p
end