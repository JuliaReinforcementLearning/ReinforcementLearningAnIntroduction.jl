using Ju
using Plots
using StatsBase:mean
using ..BlackJack
gr()

figpath(f) = "docs/src/assets/figures/figure_$f.png"

const Indices = LinearIndices(size(observationspace(BlackJackEnv)))

function preprocessor(obs) 
    Indices[CartesianIndex(obs)]
end

const player_policy = begin
    table = ones(Int, length(Indices))
    table[Indices[:, 10:11, :]] .= 2
    DeterministicPolicy(table, 2)
end

function fig_5_1(n=10000)
    agent = Agent(MonteCarloLearner(TabularV(length(Indices)), player_policy, 1.0),
                  EpisodeSARDBuffer(),
                  preprocessor)
    train!(BlackJackEnv(), agent; callbacks=(stop_at_step(n),))
    usable_ace_values = [agent.learner.approximator(preprocessor(BlackJack.encode(1, player_sum, dealer_card))) 
                         for dealer_card in 2:11, player_sum in 11:21]
    no_usable_ace_values = [agent.learner.approximator(preprocessor(BlackJack.encode(0, player_sum, dealer_card))) 
                            for dealer_card in 2:11, player_sum in 11:21]
    p1 = heatmap(usable_ace_values)
    p2 = heatmap(no_usable_ace_values)
    savefig(p1, figpath("5_1_usable_ace_n_$n"))
    savefig(p2, figpath("5_1_no_usable_ace_n_$n"))
    p1, p2
end

"TODO: WARNING!!! result is not the same with the implementation by Python"
function fig_5_2(n=1000000)
    agent = Agent(MonteCarloExploringStartLearner(TabularQ(length(Indices), length(actionspace(BlackJackEnv))),
                                        player_policy,
                                        RandomPolicy(fill(0.5, length(Indices), length(actionspace(BlackJackEnv)))),
                                        1.0;
                                        is_first_visit=false),
                  EpisodeSARDBuffer(),
                  preprocessor)
    train!(BlackJackEnv(;is_random_start=true), agent; callbacks=(stop_at_episode(n),))
    usable_ace_values = [agent.learner.approximator(preprocessor(BlackJack.encode(1, player_sum, dealer_card)), Val(:max)) 
                         for dealer_card in 2:11, player_sum in 11:21]
    usable_ace_policy = [agent.learner.π(preprocessor(BlackJack.encode(1, player_sum, dealer_card)))
                         for player_sum in 11:21, dealer_card in 2:11]
    no_usable_ace_values = [agent.learner.approximator(preprocessor(BlackJack.encode(0, player_sum, dealer_card)), Val(:max)) 
                            for dealer_card in 2:11, player_sum in 11:21]
    no_usable_ace_policy = [agent.learner.π(preprocessor(BlackJack.encode(0, player_sum, dealer_card)))
                            for player_sum in 11:21, dealer_card in 2:11]
    p1 = heatmap(usable_ace_values)
    p2 = heatmap(no_usable_ace_values)
    p3 = heatmap(usable_ace_policy)
    p4 = heatmap(no_usable_ace_policy)
    savefig(p1, figpath("5_2_usable_ace_n_$n"))
    savefig(p2, figpath("5_2_no_usable_ace_n_$n"))
    savefig(p3, figpath("5_2_usable_ace_policy_n_$n"))
    savefig(p4, figpath("5_2_no_usable_ace_policy_n_$n"))
    p1, p2, p3, p4
end

function fig_5_3(n=10000)
    init_internal_state = [1, 13, 2]
    s = preprocessor(BlackJack.encode(init_internal_state...))

    function value_collect()
        values = []
        function f(env, agent)
            if agent.buffer.isdone[end]
                push!(values, agent.learner.approximator(s))
            end
        end
        f() = values
        f
    end

    function run(sampling=:WeightedImportanceSampling)
        agent = Agent(OffPolicyMonteCarloLearner(TabularV(length(Indices)),
                                                RandomPolicy(fill(0.5, length(Indices), length(actionspace(BlackJackEnv)))),
                                                player_policy,
                                                1.0;
                                                isfirstvisit=true,
                                                sampling=sampling),
                    EpisodeSARDBuffer(),
                    preprocessor)
        callbacks = (stop_at_episode(n, false), value_collect())
        train!(BlackJackEnv(;init=init_internal_state), agent; callbacks=callbacks)
        callbacks[2]()
    end
    p = plot(mean((run() .- (-0.27726)).^2 for _ in 1:100), label="Weighted Importance Sampling")
    p = plot!(p, mean((run(:OrdinaryImportanceSampling) .- (-0.27726)).^2 for _ in 1:100), xscale=:log10, label="Ordinary Importance Sampling")
    savefig(p, figpath("5_3"))
    p
end