using Ju
using ProgressMeter
using StatsBase:mean
using ..ShortCorridor
using NNlib:softmax
using Plots
gr()

figpath(f) = "docs/src/assets/figures/figure_$f.png"

import Ju:LinearPolicy

function (π::LinearPolicy)(s, ::Val{:dist}) 
    prob = @view(π.features[s, :, :]) * π.weights |> softmax
    # force explore, or it will never reach the end of an episode
    if minimum(prob) < 0.05
        max_ind = argmax(prob)
        prob = fill(0.05/length(prob), length(prob))
        prob[max_ind] = 0.95
    end
    prob
end

function run_once_RL()
    env = ShortCorridorEnv()
    features = zeros(length(observationspace(env)), length(actionspace(env)), 2)
    for i in axes(features, 1)
        features[i, :, :] .= [0 1; 1 0]
    end
    agent = Agent(ReinforceLearner(LinearPolicy(features, [-1.47, 1.47]),
                                2e-4,
                                1.),
                EpisodeSARDBuffer())
    callbacks = (stop_at_episode(1000, false),  rewards_of_each_episode())
    train!(env, agent; callbacks=callbacks)
    callbacks[2]()
end

"Some episode may never end due. Because the policy may exploit one action"
function fig_13_1()
    p = plot(legend=:bottomright, dpi = 200)
    avg_rewards = run_once_RL()
    @showprogress for _ in 1:99
        avg_rewards .+= run_once_RL()
    end
    plot!(p, avg_rewards ./100)
    savefig(p, figpath("13_1"))
    p
end

function run_once_RLBaseline()
    env = ShortCorridorEnv()
    features = zeros(length(observationspace(env)), length(actionspace(env)), 2)
    for i in axes(features, 1)
        features[i, :, :] .= [0 1; 1 0]
    end
    agent = Agent(ReinforceBaselineLearner(TabularV(zeros(length(observationspace(env)))),
                                           LinearPolicy(features, [-1.47, 1.47]),
                                           1e-4,
                                           1e-4,
                                           1.),
                EpisodeSARDBuffer())
    callbacks = (stop_at_episode(1000, false),  rewards_of_each_episode())
    train!(env, agent; callbacks=callbacks)
    callbacks[2]()
end

function fig_13_2()
    p = plot(legend=:bottomright, dpi = 200)
    avg_rewards = run_once_RLBaseline()
    @showprogress for _ in 1:99
        avg_rewards .+= run_once_RLBaseline()
    end
    plot!(p, avg_rewards ./100, label="ReinforceBaseline")

    avg_rewards = run_once_RL()
    @showprogress for _ in 1:99
        avg_rewards .+= run_once_RL()
    end
    plot!(p, avg_rewards ./100, label="Reinforce")

    savefig(p, figpath("13_2"))
    p
end