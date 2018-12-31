using Ju
using ..TicTacToe

"DEBUG ONLY. Used to show training progress"
function gen_records()
    winners = Dict(:offensive=>0, :defensive=>0, :invalid_defensive=>0, :invalid_offensive=>0, :tie=>0)
    i = 0
    function record(env, agents)
        if isend(env)
            i += 1
            if env.state == TicTacToe.INVALID_STATE
                if env.role == :offensive
                    winners[:invalid_offensive] += 1
                elseif env.role == :defensive
                    winners[:invalid_defensive] += 1
                else
                    error("Impossible")
                end
            elseif agents[:offensive].buffer.reward[end] == 1.0
                winners[:offensive] += 1
            elseif agents[:defensive].buffer.reward[end] == 1.0
                winners[:defensive] += 1
            else
                winners[:tie] += 1
            end
        end
        if i % 10000 == 0
            println(i, winners)
            winners = Dict(:offensive=>0, :defensive=>0, :invalid_defensive=>0, :invalid_offensive=>0, :tie=>0)
        end
    end
end


"""
Yeah I know it's a little ad-hoc here.

It's not a good practice to combine the value approximator (`TabularV`) and the policy (`FunctionalPolicy`) together.
However, that means we need an environment model to tell us how to choose an action base on the value approximator.
And it will make things a little comlicate here.

Notice that here the value approximator is initialized to a very good start point.
It would take much longer time if you choose to initialize the value approximator randomly.
"""
function prepare_V_π(role)
    V = TabularV(TicTacToe.estimation_of(role))
    π = FunctionalPolicy() do s
        actions, states = TicTacToe.get_possible_action_state(s, role)
        actions[(V(s) for s in states) |> EpsilonGreedySelector(0.01)]
    end
    V, π
end

function pre_train()
    env = TicTacToeEnv()
    nstates, nactions = length(observationspace(env)), length(actionspace(env))
    V, π = prepare_V_π(:offensive)
    offensive_agent = Agent(MonteCarloLearner(V, π, 1.0, 0.1, false),
                            EpisodeSARDBuffer(),
                            identity,
                            :offensive)
    V, π = prepare_V_π(:defensive)
    defensive_agent = Agent(MonteCarloLearner(V, π, 1.0, 0.1, false),
                            EpisodeSARDBuffer(),
                            identity,
                            :defensive)
    # train!(env, (offensive_agent, defensive_agent); callbacks=(stop_at_episode(10^5),  gen_records()))  # debug
    train!(env, (offensive_agent, defensive_agent); callbacks=(stop_at_episode(10^5),))
    (offensive_agent, defensive_agent)
end

const offensive_agent, defensive_agent = pre_train()

function read_action_from_stdin()
    print("Your input:")
    input = parse(Int, readline())
    !in(input, 1:9) && error("invalid input!")
    input
end

function play()
    env = TicTacToeEnv()
    println("""You play first!
    1 4 7
    2 5 8
    3 6 9""")
    while true
        action = read_action_from_stdin()
        env(action, :offensive)
        render(env)
        obs, isdone, reward = observe(env, :offensive)
        if isdone
            if reward == 0.5
                println("Tie!")
            elseif reward == 1.0 
                println("You win!")
            else
                println("Invalid input!")
            end
            break
        end

        env(defensive_agent.learner(observe(env, :defensive).observation), :defensive)
        render(env)
        obs, isdone, reward = observe(env, :defensive)
        if isdone
            if reward == 0.5
                println("Tie!")
            elseif reward == 1.0 
                println("Your lose!")
            else
                println("You win!")
            end
            break
        end
    end
end