using Ju
using ..WindyGridWorld
using Plots
gr()

figpath(f) = "docs/src/assets/figures/figure_$f.png"

function episode_count()
    counts = []
    i = 0
    function record_episode(env, agent)
        if agent.buffer.isdone[end]
            i += 1
        end
        push!(counts, i)
    end
    record_episode() = counts
end

function gen_env_agent() 
    env = WindyGridWorldEnv()
    n_states = length(observationspace(env))
    n_actions = length(actionspace(env))
    policy = EpsilonGreedyPolicy(rand(1:n_actions, n_states), n_actions, 0.1)
    agent = Agent(TDLearner(TabularQ(n_states, n_actions), policy, 1.0, 0.5),
                  EpisodeSARDBuffer())
    env, agent
end

function fig_6_2_d()
    env, agent = gen_env_agent()
    callbacks = (stop_at_step(8000), episode_count())
    train!(env, agent; callbacks=callbacks)
    p = plot(callbacks[2](), legend=:bottomright, dpi=200)
    savefig(p, figpath("6_2_d"))
    p
end