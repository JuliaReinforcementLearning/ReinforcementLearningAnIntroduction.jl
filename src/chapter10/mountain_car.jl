using Ju
using ProgressMeter
using StatsBase:mean
using ..MountainCar:POSITION_MAX, POSITION_MIN, VELOCITY_MAX, VELOCITY_MIN, MountainCarEnv
using Plots
gr()


function cost_to_go(approximator)
    [-approximator([p, v], Val(:max))
     for p in range(POSITION_MIN, stop=POSITION_MAX, length=40),
         v in range(VELOCITY_MIN, stop=VELOCITY_MAX, length=40)]
end

function gen_env_agent(α=2e-4, n=0)
    env = MountainCarEnv()
    obs_space = observationspace(env)
    
    ntilings = 8
    ntiles = 8
    tiling = Tiling(Tuple(range(l, step=(h-l)/ntiles, length=ntiles+2) for (l, h) in zip(obs_space.low, obs_space.high)))
    offset = (obs_space.high .- obs_space.low) ./ (ntiles * ntilings)
    tilings = [tiling - offset .* (i-1) for i in 1:ntilings]
    
    nactions = length(actionspace(env))
    agent = Agent(TDLearner(TilingsQ(tilings, nactions),
                            EpsilonGreedySelector(0.),
                            1.0,
                            α,
                            n),
                    EpisodeSARDBuffer(;state_type=Vector{Float64}))
    env, agent
end

function fig_10_1()
    env, agent = gen_env_agent()
    for (i, nepisode) in enumerate([1, 11, 93, 907, 8093])
        train!(env, agent; callbacks=(stop_at_episode(nepisode),))
        p = heatmap(cost_to_go(agent.learner.approximator))
        savefig(p, "figure_10_1_$i.png")
    end
end

function fig_10_2()
    p = plot(legend=:topright, dpi = 200)
    for α in [0.1/8, 0.2/8, 0.5/8]
        avg_steps_per_episode = zeros(500)
        @showprogress for _ in 1:100
            callbacks = (stop_at_episode(500, false), steps_per_episode())
            train!(gen_env_agent(α)...; callbacks=callbacks)
            avg_steps_per_episode .+= callbacks[2]()
        end
        plot!(p, avg_steps_per_episode ./ 500)
    end
    savefig(p, "figure_10_2.png")
    p
end

function fig_10_3()
    p = plot(legend=:topright, dpi = 200)
    for (α, n) in [(0.5/8, 1), (0.3/8, 8)]
        avg_steps_per_episode = zeros(500)
        @showprogress for _ in 1:100
            callbacks = (stop_at_episode(500, false), steps_per_episode())
            train!(gen_env_agent(α, n)...; callbacks=callbacks)
            avg_steps_per_episode .+= callbacks[2]()
        end
        plot!(p, avg_steps_per_episode ./ 500)
    end
    savefig(p, "figure_10_3.png")
    p
end

function fig_10_4()
    function run_once(α, n)
        callbacks = (stop_at_episode(50, false), steps_per_episode())
        train!(gen_env_agent(α, n)...; callbacks=callbacks)
        mean(callbacks[2]())
    end

    p = plot(legend=:topright, dpi = 200)
    @showprogress for (A, n) in [(0.4:0.1:1.7, 1), (0.3:0.1:1.6, 2), (0.2:0.1:1.4, 4), (0.2:0.1:1.0, 8), (0.2:0.1:0.7, 16)]
        plot!(p, A, [mean(run_once(α/8, n) for _ in 1:5) for α in A], label="n = $n")
    end
    savefig(p, "figure_10_4.png")
    p
end