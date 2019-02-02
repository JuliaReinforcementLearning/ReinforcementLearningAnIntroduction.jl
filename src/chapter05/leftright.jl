using Ju
using ProgressMeter
using ..LeftRight
using Plots
gr()



function fig_5_4(fig_dir=".")
    function value_collect()
        values = []
        function f(env, agent)
            if agent.buffer.isdone[end]
                push!(values, agent.learner.approximator(1))
            end
        end
        f() = values
        f
    end

    p = plot(legend = false, dpi = 200)
    @showprogress for _ in 1:10
        agent = Agent(OffPolicyMonteCarloLearner(TabularV(2),
                                                RandomPolicy(fill(0.5, 2, 2)),
                                                DeterministicPolicy(ones(Int, 2), 2),
                                                1.0;
                                                isfirstvisit = true,
                                                sampling = :OrdinaryImportanceSampling),
                        EpisodeSARDBuffer())
        callbacks = (stop_at_episode(100000, false), value_collect())
        train!(LeftRightEnv(), agent; callbacks = callbacks)
        plot!(p, callbacks[2](), xscale = :log10)
    end
    savefig(p, joinpath(fig_dir, "figure_5_4.png"))
    p
end