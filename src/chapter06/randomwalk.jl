using Ju
using Statistics
using ..RandomWalk
using Plots
gr()



const true_values = [i/6 for i in 1:5]

function record_rms()
    rms = []
    function calc_rms(env, agent)
        if agent.buffer.isdone[end]
            push!(rms, sqrt(mean((agent.learner.approximator.table[2:end - 1] - true_values).^2)))
        end
    end
    calc_rms() =  rms
end

function gen_env_TDagent(α) 
    env = RandomWalkEnv(leftreward=0.)
    n_states = length(observationspace(env))
    n_actions = length(actionspace(env))
    agent = Agent(TDLearner(TabularV(fill(0.5, n_states)),
                            RandomPolicy(fill(1 / n_actions, n_states, n_actions)),
                            1.0,
                            α),
                EpisodeSARDBuffer())
    env, agent
end

"""
Here we use FirstVisitMC
"""
function gen_env_MCagent(α) 
    env = RandomWalkEnv(leftreward=0.)
    n_states = length(observationspace(env))
    n_actions = length(actionspace(env))
    agent = Agent(MonteCarloLearner(TabularV(fill(0.5, n_states)),
                                    RandomPolicy(fill(1 / n_actions, n_states, n_actions)),
                                    1.0,
                                    α,
                                    false),
                  EpisodeSARDBuffer())
    env, agent
end

function fig_6_2_a()
    p = plot(dpi = 200)
    for i in [1, 9, 90]
        env, agent = gen_env_TDagent(0.1)
        train!(env, agent; callbacks = (stop_at_episode(i),))
        plot!(p, agent.learner.approximator.table[2:end - 1])
    end
    savefig(p, "figure_6_2_a.png")
    p
end

function fig_6_2_b()
    p = plot(dpi = 200)
    for α in [0.05, 0.1, 0.15]
        callbacks = (stop_at_episode(100), record_rms())
        train!(gen_env_TDagent(α)...;callbacks = callbacks)
        plot!(p, callbacks[2](), label ="TD alpha=$α")
    end

    for α in [0.01, 0.02, 0.03, 0.04]
        callbacks = (stop_at_episode(100), record_rms())
        train!(gen_env_MCagent(α)...;callbacks = callbacks)
        plot!(p, callbacks[2](), label ="MC alpha=$α")
    end
    savefig(p, "figure_6_2_b.png")
    p
end

function fig_6_2_c()
    p = plot(dpi = 200)
    avg_rms = []
    for i in 1:100
        callbacks = (stop_at_episode(100), record_rms())
        train!(gen_env_TDagent(0.1)...;callbacks=callbacks)
        push!(avg_rms, callbacks[2]())
    end
    plot!(mean(avg_rms), color=:blue, label="TD")

    avg_rms = []
    for i in 1:100
        callbacks = (stop_at_episode(100), record_rms())
        train!(gen_env_MCagent(0.1)...;callbacks=callbacks)
        push!(avg_rms, callbacks[2]())
    end
    plot!(mean(avg_rms), color=:red, label="MC")

    savefig(p, "figure_6_2_c.png")
    p
end