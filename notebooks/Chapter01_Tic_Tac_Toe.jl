### A Pluto.jl notebook ###
# v0.16.4

using Markdown
using InteractiveUtils

# ╔═╡ 9b8c8d1a-481e-11eb-1b85-91264e100b12
begin
	import Pkg
	Pkg.activate(Base.current_project())
	using ReinforcementLearning
end

# ╔═╡ 7441759c-4853-11eb-3d63-2be1f95f59fe
using Plots

# ╔═╡ 704d34fc-4859-11eb-2d95-45c4d5246b26
begin
	using Flux
	A = TabularVApproximator(;n_state=10, init=0.0, opt=InvDecay(1.0))
	A
end

# ╔═╡ 82338e22-4805-11eb-0b54-49f734001f0c
md"""
## Introduction

In the following notebooks, we'll mainly use the [ReinforcementLearning.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl) package to demonstrate how to generate figures in the book [Reinforcement Learning: An Introduction(2nd)](http://incompleteideas.net/book/the-book-2nd.html). In case you haven't used this package before, you can visit [juliareinforcementlearning.org](https://juliareinforcementlearning.org/) for a detailed introduction. Besides, we'll also explain some basic concepts gradually when we meet them for the first time.

[ReinforcementLearning.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl) contains a collection of tools to describe and solve problems we want to handle in the reinforcement learning field. Though we are mostly interested in traditional tabular methods here, it also contains many state-of-the-art algorithms. To use it, we can simply run the following code here:
"""

# ╔═╡ bfe260f4-481e-11eb-28e8-eb3cca823162
md"""
!!! note
	As you might have noticed, it takes quite a long time to load this package for the first time (the good news is that it will be largely decreased after `Julia@v1.6`). Once loaded, things should be very fast.

"""

# ╔═╡ 147a9204-4841-11eb-04fd-4f760dff4bc8
md"""
## Tic-Tac-Toe

In chapter 1.5, a simple game of [Tic-Tac-Toe](https://en.wikipedia.org/wiki/Tic-tac-toe) is introduced to illustrate the general idea of reinforcement learning. Before looking into the details about how to implement the Monte Carlo based policy, let's take a look at how the Tic-Tac-Toe environment is defined in [ReinforcementLearning.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl).
"""

# ╔═╡ 46d5181a-4845-11eb-222c-3d8b78f89dc6
env = TicTacToeEnv()

# ╔═╡ d9a93b80-4827-11eb-39cf-c589ebcd092b
md"""
There is many important information provided above. First, the `TicTacToeEnv` is a [zero-sum](https://en.wikipedia.org/wiki/Zero-sum_game), two player environment. `Sequential` means each player takes an action alternatively. Since all the players can observe the same information (the board), it is an environment of [perfect information](https://en.wikipedia.org/wiki/Perfect_information). Note that each player only gets the reward at the end of the game (`-1`, `0`, or `1`). So we call the `RewardStyle` of this kind of environments `TerminalReward()`. At each step, only part of the actions are valid (the position of the board), so we say the `ActionStyle` of this env is of `FullActionSet()`.

All this information is provided by traits. In later chapters, we'll see how to define these traits for new customized environments. Now let's get familiar with some basic interfaces defined with the environment first.
"""

# ╔═╡ fad9dddc-4827-11eb-1b79-db762523f380
state(env) |> Text  # Here we use Text to show the string properly in Pluto

# ╔═╡ 483e774c-4849-11eb-1d48-c3ba0f63a784
md"""
By default, the state of the `TicTacToeEnv` is represented as a `String`. But some other types are also provided.

First, the number of possible valid boards is numerable. You might be interested in this [blog](http://www.occasionalenthusiast.com/tag/tic-tac-toe/) to figure out the exact number. So we can use an integer to represent the state.
"""

# ╔═╡ ef6c5598-4849-11eb-1b76-d9aff5a62abb
state(env, Observation{Int}())

# ╔═╡ f07794e2-484a-11eb-2cc8-e318bf15d469
md"""
Another way to express the state is to use a one-hot bool Array to represent the state. The first and the second dimension mean the row and column of the board. The third dimension means the role in each cell (`x`, `o` or empty). This kind of representation is usually more friendly to neural network based algorithms.
"""

# ╔═╡ b8521fe6-4837-11eb-0e53-ab179c6c3ea8
state(env, Observation{BitArray{3}}())

# ╔═╡ 9e42c688-4829-11eb-0e7a-f93d399dbf09
md"""
Each environment is a functional object which takes in an action and changes its internal state correspondingly.
"""

# ╔═╡ ff62343e-484b-11eb-2a41-29768287e2b4
begin
	env(1)
	env(2)
	env(3)
	env(4)
	env(5)
	env(6)
	env(7)
	state(env) |> Text
end

# ╔═╡ 52ac24ce-484c-11eb-25ed-bf7a4bd7f9fb
md"""
Now we can check some basic information about the `env`:
"""

# ╔═╡ 62a66722-484c-11eb-32d0-7b2695b67e6b
is_terminated(env), [reward(env, p) for p in players(env)]

# ╔═╡ 83ceeb54-484c-11eb-0499-33bafa9f569b
md"""
As you can see, the game is now terminated, and the reward of each player is $(join([reward(env, p) for p in players(env)], ", ")).
"""

# ╔═╡ c842c388-48b9-11eb-3474-b177c74c02c9
reset!(env)

# ╔═╡ cd5ddb32-48b9-11eb-0a3f-41c723469e9a
state(env) |> Text

# ╔═╡ 45d38868-484d-11eb-016a-eff04cc182a8
md"""
## RandomPolicy

Among all the policies provided in `ReinforcementLearning.jl`, the simplest one is `RandomPolicy`.
"""

# ╔═╡ 1697cfde-482d-11eb-1da9-5537e88a70c5
policy = RandomPolicy()

# ╔═╡ 177f3ee6-482b-11eb-36c3-c97e945cbd0b
[policy(env) for _ in 1:10]

# ╔═╡ 5b589dbc-482b-11eb-2395-317aca83d1f4
state(env) |> Text

# ╔═╡ 505e0fe6-482b-11eb-1df7-45ebb6c65c9b
md"""
As you can see, each policy is also a functional object which takes in a `env` and returns an action.

!!! note
	A policy will never change the internal state of an `env`. The `env` will only get updated after executing `env(action)`.
"""

# ╔═╡ 19b8290e-484e-11eb-33f3-9d2329d15a46
md"""
## Policy + Environment

Now that we have understood the two main concepts in reinforcement learning, we can roll out several simulations and collect some information we are interested in.
"""

# ╔═╡ c6045bfe-484e-11eb-2b9c-6f230a90cb03
run(policy, env, StopAfterEpisode(1))

# ╔═╡ 297a9614-484f-11eb-24a4-bd80771c1f1c
md"""
The above `run` function defined in `ReinforcementLearning.jl` is quite straight forward. The `policy` generates an action at each time step and feeds it into the `env`. The process continues until the end of an episode. Here `StopAfterEpisode(1)` is a built in stop condition. You can also see many other stop conditions in the [doc](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_core/#ReinforcementLearningCore.StopAfterEpisode).

You are encouraged to read the [source code](https://github.com/JuliaReinforcementLearning/ReinforcementLearningCore.jl/blob/master/src/core/run.jl) of this function. It's pretty simple (less than 30 lines) and easy to understand. I'll wait you here until you are finished.

If you have finished reading it, you'll notice that one important argument is missing in the above function call, the `hook`. Now we'll add the fourth argument to collect the reward of each player in every episode.
"""

# ╔═╡ 6f58815c-4852-11eb-18e8-e55b276ba228
multi_agent_policy = MultiAgentManager(
	(
		NamedPolicy(p=>RandomPolicy())
		for p in players(env)
	)...
)

# ╔═╡ 7ff2ce0a-4852-11eb-3f97-c1bc0b1bc056
multi_agent_hook = MultiAgentHook(
	(
		p => TotalRewardPerEpisode()
		for p in players(env)
	)...
)

# ╔═╡ e8e0ad64-484e-11eb-3eba-e7b323105de1
run(multi_agent_policy,	env, StopAfterEpisode(10), multi_agent_hook)

# ╔═╡ bb374d92-4852-11eb-1268-976e9e5532f4
md"""
Besides that we added a fourth argument in the `run`, another important difference is that we used a `MultiAgentManager` policy instead of a simple `RandomPolicy`. The reason is that we want different players to use a separate policy and then we can collect different information of them separately.

Now let's take a look at the total reward of each player in the above 10 episodes:
"""

# ╔═╡ 6f7bddcc-4853-11eb-2e38-edc41fcbd1ec
x, o = players(env)

# ╔═╡ 97e68c62-4853-11eb-2e19-43fc6a6a4618
begin
	plot(multi_agent_hook[x][]; label="x")
	plot!(multi_agent_hook[o][]; label="o")
end

# ╔═╡ b7ded1bc-4855-11eb-02a7-2f77d10767b3
md"""
## Tackling the Tic-Tac-Toe Problem with Monte Carlo Prediction

Actually the Monte Carlo method mentioned in the first chapter to solve the Tic-Tac-Toe problem is explained in Chapter 5.1, so it's ok if you don't fully understand it right now for the first time. Just keep reading and turn back to this chapter once you have finished chapter 5.

The intuition behind Monte Carlo prediction is that we use a table to record the estimated gain at each step. If such estimation is accurate, then we can simply look one step ahead and compare which state is of the largest gain. Then we just take the action which will lead to that state to maximize our reward.

### TabularVApproximator

In `ReinforcementLearning.jl`, the fundamental components used to estimate state values (or state-action values, or any other values we are interested in) are called *AbstractApproximator*. In this simple example, we'll use the `TabularVApproximator` to estimate the state value.
"""

# ╔═╡ 3889b408-49b4-11eb-02b6-cfd5fbb885b4
md"""
A `TabularVApproximator` uses a `Vector` underlying to estimate the state value. Here the **Tabular** means that the state is required to be an `Integer`. The code above created a `TabularVApproximator` which has a table of `10` states, `1:10`, and each state value is initialized to `0.0`. We can get the estimation by:
"""

# ╔═╡ fa9d93ac-49b4-11eb-105d-958e3031f2eb
A(1)

# ╔═╡ 0d999a3c-49b5-11eb-1b74-355c4603b71a
md"""
And we can update the estimation via:
"""

# ╔═╡ 88464c8a-49b5-11eb-3327-81e653872ecf
update!(A, 2 => A(2) - 5.)

# ╔═╡ ad7f07ca-49bb-11eb-3a96-4d649425d821
md"""
Let's spend some time on what's happening here. The `update!` method provided in `ReinforcementLearning.jl` adopts the similar interface defined in [`Flux.Optimise`](https://github.com/FluxML/Flux.jl/blob/master/src/optimise/Optimise.jl). Here the `2 => A(2) - 5.)` is something similar to `Grad`. And then we apply the optimiser (`A.optimizer`) to calculate the error we want to apply on the approximator. The default optimizer in `TabularApproximator` is `InvDecay(1.0)`. The effect is the same with calculating the mean value of all the examples.
"""

# ╔═╡ e4d58428-49bc-11eb-19aa-35759a992cac
begin
	examples = 1:10
	for x in examples
		update!(A, 1 => A(1) - x)
	end
	A(1) == (0. #= the init value is 0. =# + sum(examples)) / (1+length(examples))
end

# ╔═╡ 4c7c124a-49bd-11eb-0fa9-99fe3631af10
md"""
By using the same interfaces defined in `Flux.jl`, we get a bunch of different optimizers free. You can take a look at all the supported `optimizers` defined in [optimisers.jl](https://github.com/FluxML/Flux.jl/blob/master/src/optimise/optimisers.jl). In future notebooks, we'll see how to apply some other optimizers beyond `InvDecay`.
"""

# ╔═╡ f60a5790-49bd-11eb-16b4-df2b3a682d86
md"""
### MonteCarloLearner

In `ReinforcementLearning.jl`, many different variants of Monte Carlo learners are supported. We'll skip the implementation detail for now. But to convince you that implementing such an algorithm is quite simple and straightforward in this package, we can take a look at the code snippet and compare it with the pseudocode on the book.

```julia
function _update!(
    ::EveryVisit,
    ::TabularVApproximator,
    ::NoSampling,
    L::MonteCarloLearner,
    t::AbstractTrajectory,
)
    S, R = t[:states], t[:rewards]
    V, G, γ = L.approximator, 0.0, L.γ
    for i in length(R):-1:1
        s, r = S[i], R[i]
        G = γ * G + r
        update!(V, s => V(s) - G)
    end
end
```
"""

# ╔═╡ 266f60c2-49c0-11eb-1ad9-f1c363389e38
M = MonteCarloLearner(;
	approximator = TabularVApproximator(;n_state=5478, init=0., opt=InvDecay(1.0)),
	γ = 1.0,
	kind = FIRST_VISIT,
	sampling = NO_SAMPLING
)

# ╔═╡ 55b6eadc-49d7-11eb-2080-37fd57265468
md"""
The `MonteCarloLearner` is just a wrapper of the `TabularVApproximator` here. We can feed the `env` to it and get the the state estimation.
"""

# ╔═╡ c04e438c-49d8-11eb-3888-61e46bd24f25
M(1)

# ╔═╡ 14422efe-49dc-11eb-3c4b-81a6e239b512
M(env)

# ╔═╡ ee3ddc56-49db-11eb-2150-1dc2740b1f2f
md"""
Uh-oh! The `MethodError` above tells us that the `MonteCarloLearner` expects the input to be an `Int` but gets a `String`. The reason is that the default state style of `env` is `Observation{String}()`. So we need to set the default state style to `Observation{Int}()`.
"""

# ╔═╡ 7b979bd4-49dc-11eb-26b6-4d45d01a2301
E = DefaultStateStyleEnv{Observation{Int}()}(env)

# ╔═╡ a6176830-49dc-11eb-3e04-112e6594bfbe
state(E)

# ╔═╡ 8b87797e-49de-11eb-0f22-d79e3399bd1b
md"""
Now that we have the `MonteCarloLearner` to estimate state values, how can we use it to generate actions? A simple approach is that, we try each valid action and see what state it leads to. Then we simply select the one which results in the state of the largest state value.

!!! note
	Obviously this approach only applies to deterministic environments with a small action set.

But usually this approach is more suitable for **testing** instead of **training**. Because always selecting the best action may fall into local minima and hurt the performance. You'll see more discussions on **exploration** and **exploitation** in the next chapter. Here we use the `EpsilonGreedy` method to select action.
"""

# ╔═╡ d74aecfc-4a4e-11eb-14da-75f749a4f743
explorer = EpsilonGreedyExplorer(0.1)

# ╔═╡ fedae1de-4a4e-11eb-37ed-0d32687a9b9b
begin
	values = [1, 2, 3, 1]
	N = 1000
	actions = [explorer(values) for _ in 1:N]
	plot([sum(actions .== i)/N for i in 1:length(values)])
end

# ╔═╡ d5742096-4a4f-11eb-32da-f387bcc0ca25
md"""
The above figure shows that with $\epsilon$ set to `0.1`, we have the probability of $\epsilon$ to select an action randomly and $1-\epsilon$ to select the action of the highest value.

Below, we define a function to illustrate how to select an action given an environment.
"""

# ╔═╡ a89a0f98-4a4c-11eb-2226-1b332c23ed34
function select_action(env, V)
	A = legal_action_space(env)
    values = map(A) do a
        V(child(env, a))
    end
    A[explorer(values)]
end

# ╔═╡ dc31b262-4a50-11eb-3709-d7652d590bcd
md"""
Combining the `MonteCarloLearner` and the `select_action` function, we get our policy:
"""

# ╔═╡ 42995fe4-49e0-11eb-0a60-cfe8518ee90c
P = VBasedPolicy(;learner=M, mapping=select_action)

# ╔═╡ 752cf5ec-49e0-11eb-1440-d7ce675d8fc3
reset!(E)

# ╔═╡ 79865c8c-49e0-11eb-30b8-ddf82be2f473
P(E)

# ╔═╡ ae65a714-49e0-11eb-01d4-39bacc61d24a
md"""
That's it!

Now we have a policy and we can use it to roll our simulations.
"""

# ╔═╡ d11cc120-49e0-11eb-2588-e11f814a0548
run(P, E, StopAfterEpisode(10))

# ╔═╡ f03c55e8-49e0-11eb-2dc3-a3018c810c5e
md"""
## Training

One main question we haven't answered is, **how to train the policy?**

Well, the usage is similar to the above one, the only difference is now we wrap the `policy` in an `Agent`, which is also an `AbstractPolicy`. An `Agent` is `policy + trajectory`, or people usually call it *experience replay buffer*.
"""

# ╔═╡ d04a563a-49e1-11eb-1326-b1993ee989ec
policies = MultiAgentManager(
	(
		Agent(
			policy = NamedPolicy(
				p => VBasedPolicy(;
					learner=MonteCarloLearner(;
						approximator = TabularVApproximator(;
							n_state=length(state_space(E)),
							init=0.,
							opt=InvDecay(1.0)
						),
						γ = 1.0,
						kind = FIRST_VISIT,
						sampling = NO_SAMPLING
					),
					mapping = select_action
				)
			),
			trajectory =VectorSARTTrajectory(
					;state=Int,
					action=Union{Int, NoOp},
					reward=Int,
					terminal=Bool
			)
		)
		for p in players(E)
	)...
)

# ╔═╡ 8ed8eab4-4a51-11eb-2dfb-e5a1bf1cfac3
run(policies, E, StopAfterEpisode(100_000))

# ╔═╡ 187e6fb6-4a52-11eb-3d84-e36d0f64a0c4
md"""
The interface is almost the same with the above one. Let me explain what's happening here.

First, the `policies` is a `MultiAgentManager` and the environment is a `Sequential` one. So at each step, the agent manager forwards the `env` to its inner policies. Here each inner policy is an `Agent`. Then it will collect necessary information at each step (here we used a `VectorSARTTrajectory` to tell the agent to collect `state`, `action`, `reward`, and `is_terminated` info). Finally, after each episode, the agent sends the `trajectory` to the inner `VBasedPolicy`, and then forwards it to the `MonteCarloLearner` to update the `TabularVApproximator`. Thanks to **multiple dispatch** each step above is fully customizable.
"""

# ╔═╡ 52392820-4a54-11eb-03eb-cf339de8e3dc
md"""
## Testing

Now we have our policy trained. We can happily run several self-plays and see the result. The only necessary change is to drop out the `Agent` wrapper.
"""

# ╔═╡ d5662b30-4a54-11eb-3ec6-23f78e2437b1
test_policies = MultiAgentManager([p.policy for (x, p) in (policies.agents)]...)

# ╔═╡ ff86531e-4a55-11eb-2b13-f50d077ec7b0
md"""
Here we use a hook to collect the reward of each episode for every player:
"""

# ╔═╡ 15c37092-4a56-11eb-26a3-716552d8deba
hook = MultiAgentHook(
	(
		p => TotalRewardPerEpisode()
		for p in players(E)
	)...
)	

# ╔═╡ e225bf54-4a55-11eb-0ba6-f3854fc816a2
run(test_policies, E, StopAfterEpisode(100), hook)

# ╔═╡ 33261234-4a56-11eb-28b0-f5067e4c948e
begin
	plot(hook[x][]; label="x")
	plot!(hook[o][]; label="o")
end

# ╔═╡ 696c441c-4a56-11eb-39c0-e1d1465ca4f7
md"""
As you can see, in most cases, we reach a **tie**. But why are there still some cases that don't reach a tie?

Is it because we have not trained for enough time?

*Maybe*. But a more possible reason is that we're still using the `EpsilonGreedyExplorer` in the testing mode. 

I leave it as an exercise to change the `select_action` to a greedy version and confirm that our trained policy reaches a tie everytime.
"""

# ╔═╡ Cell order:
# ╟─82338e22-4805-11eb-0b54-49f734001f0c
# ╠═9b8c8d1a-481e-11eb-1b85-91264e100b12
# ╟─bfe260f4-481e-11eb-28e8-eb3cca823162
# ╟─147a9204-4841-11eb-04fd-4f760dff4bc8
# ╠═46d5181a-4845-11eb-222c-3d8b78f89dc6
# ╟─d9a93b80-4827-11eb-39cf-c589ebcd092b
# ╠═fad9dddc-4827-11eb-1b79-db762523f380
# ╟─483e774c-4849-11eb-1d48-c3ba0f63a784
# ╠═ef6c5598-4849-11eb-1b76-d9aff5a62abb
# ╟─f07794e2-484a-11eb-2cc8-e318bf15d469
# ╠═b8521fe6-4837-11eb-0e53-ab179c6c3ea8
# ╟─9e42c688-4829-11eb-0e7a-f93d399dbf09
# ╠═ff62343e-484b-11eb-2a41-29768287e2b4
# ╟─52ac24ce-484c-11eb-25ed-bf7a4bd7f9fb
# ╠═62a66722-484c-11eb-32d0-7b2695b67e6b
# ╟─83ceeb54-484c-11eb-0499-33bafa9f569b
# ╠═c842c388-48b9-11eb-3474-b177c74c02c9
# ╠═cd5ddb32-48b9-11eb-0a3f-41c723469e9a
# ╟─45d38868-484d-11eb-016a-eff04cc182a8
# ╠═1697cfde-482d-11eb-1da9-5537e88a70c5
# ╠═177f3ee6-482b-11eb-36c3-c97e945cbd0b
# ╠═5b589dbc-482b-11eb-2395-317aca83d1f4
# ╟─505e0fe6-482b-11eb-1df7-45ebb6c65c9b
# ╟─19b8290e-484e-11eb-33f3-9d2329d15a46
# ╠═c6045bfe-484e-11eb-2b9c-6f230a90cb03
# ╟─297a9614-484f-11eb-24a4-bd80771c1f1c
# ╠═6f58815c-4852-11eb-18e8-e55b276ba228
# ╠═7ff2ce0a-4852-11eb-3f97-c1bc0b1bc056
# ╠═e8e0ad64-484e-11eb-3eba-e7b323105de1
# ╟─bb374d92-4852-11eb-1268-976e9e5532f4
# ╠═6f7bddcc-4853-11eb-2e38-edc41fcbd1ec
# ╠═7441759c-4853-11eb-3d63-2be1f95f59fe
# ╠═97e68c62-4853-11eb-2e19-43fc6a6a4618
# ╟─b7ded1bc-4855-11eb-02a7-2f77d10767b3
# ╠═704d34fc-4859-11eb-2d95-45c4d5246b26
# ╟─3889b408-49b4-11eb-02b6-cfd5fbb885b4
# ╠═fa9d93ac-49b4-11eb-105d-958e3031f2eb
# ╟─0d999a3c-49b5-11eb-1b74-355c4603b71a
# ╠═88464c8a-49b5-11eb-3327-81e653872ecf
# ╟─ad7f07ca-49bb-11eb-3a96-4d649425d821
# ╠═e4d58428-49bc-11eb-19aa-35759a992cac
# ╟─4c7c124a-49bd-11eb-0fa9-99fe3631af10
# ╟─f60a5790-49bd-11eb-16b4-df2b3a682d86
# ╠═266f60c2-49c0-11eb-1ad9-f1c363389e38
# ╟─55b6eadc-49d7-11eb-2080-37fd57265468
# ╠═c04e438c-49d8-11eb-3888-61e46bd24f25
# ╠═14422efe-49dc-11eb-3c4b-81a6e239b512
# ╟─ee3ddc56-49db-11eb-2150-1dc2740b1f2f
# ╠═7b979bd4-49dc-11eb-26b6-4d45d01a2301
# ╠═a6176830-49dc-11eb-3e04-112e6594bfbe
# ╟─8b87797e-49de-11eb-0f22-d79e3399bd1b
# ╠═d74aecfc-4a4e-11eb-14da-75f749a4f743
# ╠═fedae1de-4a4e-11eb-37ed-0d32687a9b9b
# ╟─d5742096-4a4f-11eb-32da-f387bcc0ca25
# ╠═a89a0f98-4a4c-11eb-2226-1b332c23ed34
# ╟─dc31b262-4a50-11eb-3709-d7652d590bcd
# ╠═42995fe4-49e0-11eb-0a60-cfe8518ee90c
# ╠═752cf5ec-49e0-11eb-1440-d7ce675d8fc3
# ╠═79865c8c-49e0-11eb-30b8-ddf82be2f473
# ╟─ae65a714-49e0-11eb-01d4-39bacc61d24a
# ╠═d11cc120-49e0-11eb-2588-e11f814a0548
# ╟─f03c55e8-49e0-11eb-2dc3-a3018c810c5e
# ╠═d04a563a-49e1-11eb-1326-b1993ee989ec
# ╠═8ed8eab4-4a51-11eb-2dfb-e5a1bf1cfac3
# ╟─187e6fb6-4a52-11eb-3d84-e36d0f64a0c4
# ╟─52392820-4a54-11eb-03eb-cf339de8e3dc
# ╠═d5662b30-4a54-11eb-3ec6-23f78e2437b1
# ╟─ff86531e-4a55-11eb-2b13-f50d077ec7b0
# ╠═15c37092-4a56-11eb-26a3-716552d8deba
# ╠═e225bf54-4a55-11eb-0ba6-f3854fc816a2
# ╠═33261234-4a56-11eb-28b0-f5067e4c948e
# ╟─696c441c-4a56-11eb-39c0-e1d1465ca4f7
