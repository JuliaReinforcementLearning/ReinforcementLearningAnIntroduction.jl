using Ju
using ..BairdCounter
using Plots
gr()




function record_weights()
    weights = []
    function record(env, agent)
        push!(weights, deepcopy(agent.learner.approximator.weights))
    end
    record() = weights
end

function fig_11_2(fig_dir=".")
    env = BairdCounterEnv()
    nstates = length(observationspace(env))
    nactions = length(actionspace(env))
    init_weights = ones(Int, 8)
    init_weights[7] = 10
    features = zeros(nstates, length(init_weights))
    for i in 1:6
        features[i, i] = 2
        features[i, 8] = 1
    end
    features[7, 7] = 1
    features[7, 8] = 2

    agent = Agent(OffPolicyTDLearner(LinearV(features, init_weights),
                                    RandomPolicy([6/7, 1/7]),
                                    RandomPolicy([0., 1.]),
                                    0.99,
                                    0.01,
                                    1),
                EpisodeSARDBuffer())
    callbacks = (stop_at_step(1000), record_weights())
    train!(env, agent; callbacks=callbacks)
    weights = callbacks[2]()
    p = plot(legend=:topleft, dpi = 200)
    for i in 1:length(init_weights)
        plot!(p, [w[i] for w in weights])
    end
    savefig(p, joinpath(fig_dir, "figure_11_2.png"))
    p
end