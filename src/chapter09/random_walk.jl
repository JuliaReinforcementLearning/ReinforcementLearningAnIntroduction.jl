using Ju
using ..RandomWalk
using StatsBase:mean
using Plots
gr()



const ACTIONS = collect(Iterators.flatten((-100:-1, 1:100)))
const NS = 1002
const NA = length(ACTIONS)

function group_mapping(ngroups)
    n_per_group = div(NS, ngroups)
    function group_mapping(x)
        if x == 1
            1
        elseif x == NS
            ngroups + 2
        else
            div(x - 2, n_per_group) + 2
        end
    end
end

function count_states()
    counts = zeros(Int, NS)
    function f(env, agent)
        counts[agent.buffer.state[end]] += 1
    end
    f() = counts
end

const TRUE_STATE_VALUES = begin
    env = RandomWalkEnv(N=NS, actions=ACTIONS)
    agent = Agent(TDLearner(TabularV(NS), RandomPolicy(fill(1/NA, NA)), 1., 1e-2), EpisodeSARDBuffer())
    train!(env, agent;callbacks=(stop_at_episode(10^5),))
    agent.learner.approximator.table
end

function record_rms()
    rms = []
    function calc_rms(env, agent)
        if agent.buffer.isdone[end]
            push!(rms, sqrt(mean((agent.learner.approximator.(agent.preprocessor.(2:(NS-1))) - TRUE_STATE_VALUES[2:end-1]).^2)))
        end
    end
    calc_rms() = rms
end

function fig_9_1()
    ngroups = 10
    env = RandomWalkEnv(N=NS, actions=ACTIONS)
    agent = Agent(MonteCarloLearner(AggregationV(zeros(ngroups+2), group_mapping(ngroups)),
                                    RandomPolicy(fill(1/NA, NA)),
                                    1.0,
                                    2e-5),
                EpisodeSARDBuffer())
    callbacks = (stop_at_episode(10^5), count_states())
    train!(env, agent;callbacks=callbacks)
    p1 = plot(2:(NS-1), agent.learner.approximator.(2:(NS-1)), label="group estimation")
    plot!(p1, 2:(NS-1), TRUE_STATE_VALUES[2:(NS-1)], label="true values")
    savefig(p1, "figure_9_1_a.png")
    distribution = callbacks[2]()
    p2 = plot(distribution ./ sum(distribution), label="state distribution")
    savefig(p2, "figure_9_1_b.png")
    p1, p2
end

function fig_9_2_a()
    ngroups = 10
    env = RandomWalkEnv(N=NS, actions=ACTIONS)
    agent = Agent(TDLearner(AggregationV(zeros(ngroups+2), group_mapping(ngroups)),
                            RandomPolicy(fill(1/NA, NA)),
                            1.0,
                            2e-4),
                  EpisodeSARDBuffer())
    train!(env, agent; callbacks=(stop_at_episode(10^5),))
    p = plot(2:(NS-1), agent.learner.approximator.(2:(NS-1)), label="Approximate TD value")
    plot!(p, 2:(NS-1), TRUE_STATE_VALUES[2:(NS-1)], label="true values")
    savefig(p, "figure_9_2_a.png")
    p
end

function fig_9_2_b()
    ngroups = 20
    function run_once(n, α)
        env = RandomWalkEnv(N=NS, actions=ACTIONS)
        agent = Agent(TDLearner(AggregationV(zeros(ngroups+2), group_mapping(ngroups)),
                                RandomPolicy(fill(1/NA, NA)),
                                1.0,
                                α,
                                n),
                      EpisodeSARDBuffer())
        callbacks = (stop_at_episode(10), record_rms())
        train!(env, agent; callbacks=callbacks)
        mean(callbacks[2]())
    end
    A = 0:0.1:1
    p = plot(legend=:topleft, dpi = 200)
    for n in [2^i for i in 0:9]
        plot!(p, A, mean([run_once(n, α) for α in A] for _ in 1:100), label="n = $n")
    end
    savefig(p, "figure_9_2_b.png")
    p
end

function fig_9_5()
    function run_once_MC(approx, α)
        env = RandomWalkEnv(N=NS, actions=ACTIONS)
        agent = Agent(MonteCarloLearner(approx,
                                        RandomPolicy(fill(1/NA, NA)),
                                        1.0,
                                        α),
                    EpisodeSARDBuffer(;state_type=Float64),
                    s -> s / NS)
        callbacks = (stop_at_episode(5000), record_rms())
        train!(env, agent; callbacks=callbacks)
        callbacks[2]()
    end
    p = plot(legend=:topright, dpi = 200)
    for order in [5, 10, 20]
        plot!(p, mean(run_once_MC(FourierV(order), 0.00005) for _ in 1:1), label="Fourier $order")
        plot!(p, mean(run_once_MC(PolynomialV(order), 0.0001) for _ in 1:1), label="Polynomial $order")
    end
    savefig(p, "figure_9_5.png")
    p
end

function fig_9_10()
    function run_once(approx, α)
        env = RandomWalkEnv(N=NS, actions=ACTIONS)
        agent = Agent(MonteCarloLearner(approx,
                                        RandomPolicy(fill(1/NA, NA)),
                                        1.0,
                                        α),
                    EpisodeSARDBuffer())
        callbacks = (stop_at_episode(10000), record_rms())
        train!(env, agent; callbacks=callbacks)
        callbacks[2]()
    end
    ngroups = 5
    agg() = AggregationV(zeros(ngroups+2), group_mapping(ngroups))
    tilings() = TilingsV([Tiling((range(1-4*(i-1), step=200, length=7),)) for i in 1:50])

    p = plot(legend=:topleft, dpi = 200)
    plot!(p, mean(run_once(agg(), 1e-4) for _ in 1:10), label="aggregation")
    plot!(p, mean(run_once(tilings(), 1e-4/50) for _ in 1:10), label="tilings")
    savefig(p, "figure_9_10.png")
    p
end