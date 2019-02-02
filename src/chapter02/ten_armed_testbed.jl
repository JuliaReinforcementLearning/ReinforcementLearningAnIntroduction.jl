using Ju
using ..MultiArmBandits
using Statistics
using Plots
gr()



function collect_best_actions()
    isbest = Vector{Bool}()
    f(env, agent) = begin 
        push!(isbest, env.isbest)
    end
    f() = isbest
    f
end

function bandit_testbed(learner, truevalue=0.0)
    env = MultiArmBanditsEnv(truevalue)
    agent = Agent(learner, EpisodeSARDBuffer())
    callbacks = (stop_at_step(1000), collect_best_actions())
    train!(env, agent; callbacks=callbacks)
    agent.buffer.reward, callbacks[2]()
end

##############################

# function fig_2_1(fig_dir=".")
#     env = MultiArmBanditsEnv()
#     f = render(env)
#     savefig(f, joinpath(fig_dir, "figure_2_1.png"))
#     f
# end


function fig_2_2(fig_dir=".")
    learner(ϵ) = QLearner(TabularQ(1, 10), EpsilonGreedySelector(ϵ), 0., cached_inverse_decay())
    p = plot(layout=(2, 1), dpi=200)
    for ϵ in [0.1, 0.01, 0.0]
        stats = [bandit_testbed(learner(ϵ)) for _ in 1:2000]
        plot!(p, mean(x[1] for x in stats), subplot=1, legend=:bottomright, label="epsilon=$ϵ")
        plot!(p, mean(x[2] for x in stats), subplot=2, legend=:bottomright, label="epsilon=$ϵ")
    end
    savefig(p, joinpath(fig_dir, "figure_2_2.png"))
    p
end

function fig_2_3(fig_dir=".")
    learner1() = QLearner(TabularQ(1, 10, 5.), EpsilonGreedySelector(0.0), 0., 0.1)
    learner2() = QLearner(TabularQ(1, 10), EpsilonGreedySelector(0.1), 0., 0.1)
    p = plot(legend=:bottomright, dpi=200)
    plot!(p, mean(bandit_testbed(learner1())[2] for _ in 1:2000), label="Q_1=5, epsilon=0.")
    plot!(p, mean(bandit_testbed(learner2())[2] for _ in 1:2000), label="Q_1=0, epsilon=0.1")
    savefig(p, joinpath(fig_dir, "figure_2_3.png"))
    p
end

function fig_2_4(fig_dir=".")
    learner1() = QLearner(TabularQ(1, 10), UpperConfidenceBound(10), 0., 0.1)
    learner2() = QLearner(TabularQ(1, 10), EpsilonGreedySelector(0.1), 0., 0.1)
    p = plot(legend=:bottomright, dpi=200)
    plot!(p, mean(bandit_testbed(learner1())[1] for _ in 1:2000), label="UpperConfidenceBound, c=2")
    plot!(p, mean(bandit_testbed(learner2())[1] for _ in 1:2000), label="epsilon-greedy, epsilon=0.1")
    savefig(p, joinpath(fig_dir, "figure_2_4.png"))
    p
end

function fig_2_5(fig_dir=".")
    learner(alpha, baseline) = GradientBanditLearner(TabularQ(1, 10), WeightedSample(), alpha, baseline)
    truevalue = 4.0
    p = plot(legend=:bottomright, dpi=200)
    plot!(p, mean(bandit_testbed(learner(0.1, sample_avg()), truevalue)[2] for _ in 1:2000), label="alpha = 0.1, with baseline")
    plot!(p, mean(bandit_testbed(learner(0.4, sample_avg()), truevalue)[2] for _ in 1:2000), label="alpha = 0.4, with baseline")
    plot!(p, mean(bandit_testbed(learner(0.1, 0.), truevalue)[2] for _ in 1:2000), label="alpha = 0.1, without baseline")
    plot!(p, mean(bandit_testbed(learner(0.4, 0.), truevalue)[2] for _ in 1:2000), label="alpha = 0.4, without baseline")
    savefig(p, joinpath(fig_dir, "figure_2_5.png"))
    p
end

function fig_2_6(fig_dir=".")
    ϵ_greedy_learner(ϵ) = QLearner(TabularQ(1, 10), EpsilonGreedySelector(ϵ), 0., cached_inverse_decay())
    gradient_learner(alpha) = GradientBanditLearner(TabularQ(1, 10), WeightedSample(), alpha, sample_avg())
    UpperConfidenceBound_learner(c) = QLearner(TabularQ(1, 10), UpperConfidenceBound(10, c), 0., cached_inverse_decay())
    greedy_with_init_learner(init) = QLearner(TabularQ(1,10, init), EpsilonGreedySelector(0.), 0., 0.1)

    p = plot(legend=:bottomright, dpi=200)
    plot!(p, -7:-2, [mean(mean(bandit_testbed(ϵ_greedy_learner(2.0^i))[1] for _ in 1:2000)) for i in -7:-2], label="epsilon greedy")
    plot!(p, -5:1, [mean(mean(bandit_testbed(gradient_learner(2.0^i))[1] for _ in 1:2000)) for i in -5:1], label="gradient")
    plot!(p, -4:2, [mean(mean(bandit_testbed(UpperConfidenceBound_learner(2.0^i))[1] for _ in 1:2000)) for i in -4:2], label="UCB")
    plot!(p, -2:2, [mean(mean(bandit_testbed(greedy_with_init_learner(2.0^i))[1] for _ in 1:2000)) for i in -2:2], label="greedy with initialization")
    savefig(p, joinpath(fig_dir, "figure_2_6.png"))
    p
end