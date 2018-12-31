using Ju
using ProgressMeter
using StatsBase:mean
using ..RandomWalk
using Plots
gr()

figpath(f) = "docs/src/assets/figures/figure_$f.png"

const N = 21

const true_values = -1:0.1:1

function record_rms()
    rms = []
    function calc_rms(env, agent)
        if agent.buffer.isdone[end]
            push!(rms, sqrt(mean((agent.learner.approximator.table[2:end-1] - true_values[2:end-1]).^2)))
        end
    end
    calc_rms() =  mean(rms)
end

function gen_env_TDagent(α, n) 
    env = RandomWalkEnv(N=21)
    n_states = length(observationspace(env))
    n_actions = length(actionspace(env))
    agent = Agent(TDLearner(TabularV(fill(0., n_states)),
                            RandomPolicy(fill(1 / n_actions, n_states, n_actions)),
                            1.0,
                            α,
                            n),
                EpisodeSARDBuffer())
    env, agent
end

const A = 0.:0.05:1.0

function records(n)
    avg_rms = []
    for α in A
        rms = []
        for _ in 1:100
            callbacks = (stop_at_episode(10, false), record_rms())
            train!(gen_env_TDagent(α, n)...; callbacks=callbacks)
            push!(rms, callbacks[2]())
        end
        push!(avg_rms, mean(rms))
    end
    avg_rms
end

function fig_7_2()
    p = plot(legend=:topright, dpi = 200)
    @showprogress for n in [2^i for i in 0:9]
        plot!(p, A, records(n), label="n=$n")
    end
    savefig(p, figpath("7_2"))
    p
end