using Ju
using ProgressMeter
using StatsBase:mean
using ..RandomWalk
using Plots
gr()




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

function gen_env_TDλReturnAgent(α, λ) 
    env = RandomWalkEnv(N=21)
    n_states = length(observationspace(env))
    n_actions = length(actionspace(env))
    agent = Agent(TDλReturnLearner(TabularV(fill(0., n_states)),
                            RandomPolicy(fill(1 / n_actions, n_actions)),
                            1.0,
                            α,
                            λ),
                EpisodeSARDBuffer())
    env, agent
end

function records(α, λ, nruns=10)
    rms = []
    for _ in 1:nruns
        callbacks = (stop_at_episode(10, false), record_rms())
        env, agent = gen_env_TDλReturnAgent(α, λ)
        train!(env, agent;callbacks=callbacks)
        push!(rms, callbacks[2]())
    end
    mean(rms)
end

function fig_12_3(fig_dir=".")
    As = [0:0.1:1, 0:0.1:1, 0:0.1:1, 0:0.1:1, 0:0.1:1, 0:0.05:0.5, 0:0.02:0.2, 0:0.01:0.1]
    Λ = [0., 0.4, .8, 0.9, 0.95, 0.975, 0.99, 1.]
    p = plot(legend=:topright, dpi = 200)
    @showprogress for (A, λ) in zip(As, Λ)
        plot!(p, A, [records(α, λ) for α in A], label="lambda = $λ")
    end
    ylims!(p, (0.25, 0.55))
    savefig(p, joinpath(fig_dir, "figure_12_3.png"))
    p
end