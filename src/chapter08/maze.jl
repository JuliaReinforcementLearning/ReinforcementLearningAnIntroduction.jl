using Ju
using ..Maze
using Random
using StatsBase:mean
using Plots
gr()



function record_steps(n=0)
    env = MazeEnv()
    n_states = length(observationspace(env))
    n_actions = length(actionspace(env))
    agent = DynaAgent(QLearner(TabularQ(n_states, n_actions),
                            EpsilonGreedyPolicy(rand(1:n_actions, n_states), n_actions, 0.1),
                            0.95,
                            0.1),
                      EpisodeSARDBuffer(),
                      ExperienceSampleModel(),
                      n)
    env, agent
    callbacks = (stop_at_episode(50), steps_per_episode())
    train!(env, agent;callbacks=callbacks)
    callbacks[2]()
end

function fig_8_2(fig_dir=".")
    p = plot(legend=:topright, dpi=200)
    plot!(p, mean(record_steps(0) for _ in 1:30), label="0 planning step")
    plot!(p, mean(record_steps(5) for _ in 1:30), label="5 planning step")
    plot!(p, mean(record_steps(50) for _ in 1:30), label="50 planning step")
    savefig(p, joinpath(fig_dir, "figure_8_2.png"))
    p
end

function sum_of_rewards()
    rewards = [0]
    function acc(env, agent)
        push!(rewards, rewards[end] + agent.buffer.reward[end])
    end
    acc() = rewards
end

function cumulative_dyna_reward(model, walls, nstep1, change, nstep2)
    env = MazeEnv(walls,
                  CartesianIndex(6, 4),
                  CartesianIndex(6, 4),
                  CartesianIndex(1, 9),
                  6,
                  9)
    n_states = length(observationspace(env))
    n_actions = length(actionspace(env))
    agent = DynaAgent(QLearner(TabularQ(n_states, n_actions),
                            EpsilonGreedyPolicy(rand(1:n_actions, n_states), n_actions, 0.1),
                            0.95,
                            1.),
                      EpisodeSARDBuffer(),
                      model,
                      10)
    record_fun = sum_of_rewards()
    train!(env, agent;callbacks=(stop_at_step(nstep1), record_fun))
    change(env.walls)
    train!(env, agent;callbacks=(stop_at_step(nstep2), record_fun))
    record_fun()
end

function fig_8_4(fig_dir=".")
    walls() = Set([CartesianIndex(4, j) for j in 1:8])
    function change_walls(walls)
        pop!(walls, CartesianIndex(4,1))
        push!(walls, CartesianIndex(4,9))
    end
    p = plot(legend=:topleft, dpi=200)
    plot!(p, mean(cumulative_dyna_reward(ExperienceSampleModel(), walls(), 1000, change_walls, 2000) for _ in 1:30), label="dyna-Q")
    plot!(p, mean(cumulative_dyna_reward(TimeBasedSampleModel(4), walls(), 1000, change_walls, 2000) for _ in 1:30), label="dyna-Q+")
    savefig(p, joinpath(fig_dir, "figure_8_4.png"))
    p
end

function fig_8_5(fig_dir=".")
    walls() = Set([CartesianIndex(4, j) for j in 2:9])
    function change_walls(walls)
        pop!(walls, CartesianIndex(4,9))
    end
    p = plot(legend=:topleft, dpi=200)
    plot!(p, mean(cumulative_dyna_reward(ExperienceSampleModel(), walls(), 3000, change_walls, 3000) for _ in 1:30), label="dyna-Q")
    plot!(p, mean(cumulative_dyna_reward(TimeBasedSampleModel(4, 1e-3), walls(), 3000, change_walls, 3000) for _ in 1:30), label="dyna-Q+")
    savefig(p, joinpath(fig_dir, "figure_8_5.png"))
    p
end

function fig_8_4_example(fig_dir=".")
    function stop_le(n)
        steps = []
        count = 0
        function acc(env, agent)
            count += 1
            if agent.buffer.isdone[end]
                push!(steps, count)
                count = 0
                steps[end] <= n
            end
        end
        function acc()
            steps
        end
    end

    function run_once(model, ratio=1)
        env = MazeEnv() * ratio
        n_states = length(observationspace(env))
        n_actions = length(actionspace(env))
        agent = DynaAgent(QLearner(TabularQ(n_states, n_actions),
                                EpsilonGreedyPolicy(rand(1:n_actions, n_states), n_actions, 0.1),
                                0.95,
                                0.5),
                        EpisodeSARDBuffer(),
                        model,
                        5)
        env, agent
        train!(env, agent;callbacks=(stop_le(14 * ratio * 1.2),))
        model.sample_count
    end
    p = plot(legend=:topleft, dpi=200)
    plot!(mean([run_once(ExperienceSampleModel(), ratio) for ratio in 1:6] for _ in 1:5), label="Dyna", yscale=:log10)
    plot!(mean([run_once(PrioritizedSweepingSampleModel(), ratio) for ratio in 1:6] for _ in 1:5), label="Prioritized", yscale=:log10)
    savefig(p, joinpath(fig_dir, "figure_8_4_example.png"))
    p
end