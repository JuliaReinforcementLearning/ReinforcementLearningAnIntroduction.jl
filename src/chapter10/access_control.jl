using Ju
using ..AccessControl

using Plots
gr()



function fig_10_5()
    env = AccessControlEnv()
    nstates = length(observationspace(env))
    nactions = length(actionspace(env))
    agent = Agent(DifferentialTDLearner(TabularQ(nstates, nactions),
                                EpsilonGreedySelector(0.1),
                                0.01,
                                0.01,
                                0.,
                                1),
                EpisodeSARDBuffer())
    train!(env, agent;callbacks=(stop_at_step(2*10^6),))
    p = plot(legend=:bottomright, dpi = 200)
    for i in 1:length(AccessControl.PRIORITIES)
        plot!([agent.learner.approximator(AccessControl.TRANSFORMER[(CartesianIndex(n+1, i))], Val(:max))
               for n in 1:AccessControl.N_SERVERS],
                label="priority = $(AccessControl.PRIORITIES[i])")
    end
    savefig(p, "figure_10_5.png")
    p
end