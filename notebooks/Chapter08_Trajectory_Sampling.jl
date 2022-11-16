### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 4b149ad2-5bca-11eb-0104-0be8dd786177
begin
	using ReinforcementLearning
	using Flux
	using Statistics
	using Plots
end

# ╔═╡ 48977b2e-5bc9-11eb-2ad7-958ca5fd2ecb
md"""
# Chapter 8.6 Trajectory Sampling

The general function `run(policy, env, stop_condition, hook)` is very flexible and powerful. However, we are not restricted to use it only. In this notebook, we'll see how to use part of the components provided in `ReinforcementLearning.jl` to finish some specific experiments.

First, let's define the environment mentioned in **Chapter 8.6**:
"""

# ╔═╡ 6bf4f104-5bca-11eb-2b4a-69c4ba7ffadc
begin
	mutable struct TestEnv <: AbstractEnv
		transitions::Array{Int, 3}
		rewards::Array{Float64, 3}
		reward_table::Array{Float64, 2}
		terminate_prob::Float64
		# cache
		s_init::Int
		s::Int
		reward::Float64
		is_terminated::Bool
	end

	function TestEnv(;ns=1000, na=2, b=1, terminate_prob=0.1,init_state=1)
		transitions = rand(1:ns, b, na, ns)
		rewards = randn(b, na, ns)
		reward_table = randn(na, ns)
		TestEnv(
			transitions,
			rewards,
			reward_table, 
			terminate_prob,
			init_state,
			init_state,
			0.,
			false
		)
	end

	function (env::TestEnv)(a::Int)
		t = rand() < 0.1
		bᵢ = rand(axes(env.transitions, 1))

		env.is_terminated = t
		if t
			env.reward = env.reward_table[a, env.s]
		else
			env.reward = env.rewards[bᵢ, a, env.s]
		end

		env.s = env.transitions[bᵢ, a, env.s]

	end

	RLBase.state_space(env::TestEnv) = Base.OneTo(1:size(env.rewards, 3))
	RLBase.action_space(env::TestEnv) = Base.OneTo(1:size(env.rewards, 2))

	function RLBase.reset!(env::TestEnv)
		env.s = env.s_init
		env.reward = 0.0
		env.is_terminated = false
	end

	RLBase.is_terminated(env::TestEnv) = env.is_terminated
	RLBase.state(env::TestEnv) = env.s
	RLBase.reward(env::TestEnv) = env.reward
end

# ╔═╡ 989f834a-5bca-11eb-363a-5f4afce32520
md"""
Note that this environment is not described very clearly on the book. Part of the information are inferred from the [lisp source code](http://incompleteideas.net/book/code/sampling2.lisp).

!!! info
	Actually the lisp code is also not perfect, I spent a whole afternoon to figure out the code logic. So good luck if you also want to understand it.

The definitions above are just like any other environment we've defined before in previous chapters. Now we'll add an extra function to make it work for our **planning** purpose.
"""

# ╔═╡ 97be6cb8-5bcb-11eb-3dc7-1b8105b206d7
"""
Return all the possible next states and corresponding reward.
"""
function successors(env::TestEnv, s, a)
    S = @view env.transitions[:, a, s]
    R = @view env.rewards[:, a, s]
    zip(R, S)
end

# ╔═╡ 0c52737a-5bcd-11eb-1546-21885703f303
γ = 0.9

# ╔═╡ aa061636-5bce-11eb-1381-bb00fb913f9b
n_sweep=10

# ╔═╡ a466db90-5bcc-11eb-3385-e5117be33410
"""
Here we are only interested in the performance of Q
with env starting at state `1`. Note here we're calculating
the discounted reward.
"""
function eval_Q(Q, env;n_eval=100)
	R = 0.
	for _ in 1:n_eval
		reset!(env)
		i = 0
		while !is_terminated(env)
			a = Q(state(env)) |> argmax  # greedy
			env(a)
			R += reward(env) * γ^i
			i += 1
		end
	end
	R/n_eval
end

# ╔═╡ 9bf4aea2-5bcd-11eb-230f-cf63a009a0f9
"""
Calculate the expected gain.
"""
function gain(Q,env,s,a)
    p = env.terminate_prob
    r = env.reward_table[a, s]
    p * r + (1-p) * mean(r̄ + γ * maximum(Q(s′)) for (r̄, s′) in  successors(env, s, a))
end

# ╔═╡ b92ba1e2-5bcb-11eb-16cf-f15929c2697b
function sweep(;b = 1, ns=1000)

    na = 2
    
    α=1.0
    p = 0.1

    env= TestEnv(;ns=ns, na=na, b=b, terminate_prob=p)
    Q = TabularQApproximator(;n_state=ns, n_action=na, opt=Descent(α))

    i = 1
    vals = [eval_Q(Q, env)]
    for _ in 1:n_sweep
        for s in 1:ns
            for a in 1:na
                G = gain(Q,env,s,a)
                update!(Q, (s,a) =>  Q(s, a) - G)
                if i % 100 == 0
                    push!(vals, eval_Q(Q, env))
                end
                i += 1
            end
        end
    end
    vals
end

# ╔═╡ 86538552-5bce-11eb-2cbf-ff6e175ef421
function on_policy(;b = 1, ns=1000)

    na = 2
    
    α=1.0
    p = 0.1

    env= TestEnv(;ns=ns, na=na, b=b, terminate_prob=p)
    Q = TabularQApproximator(;n_state=ns, n_action=na, opt=Descent(α))

    i = 1
    vals = [eval_Q(Q, env)]
	
	explorer = EpsilonGreedyExplorer(0.1)
	for i in 1:(n_sweep * ns * na)
		is_terminated(env) && reset!(env)
		s = state(env)
		a = Q(s) |> explorer
		env(a)
		G = gain(Q, env, s, a)
		update!(Q, (s,a) => Q(s,a) - G)
		if i % 100 == 0
			push!(vals, eval_Q(Q, env))
		end
	end
	
    vals
end

# ╔═╡ f29c9f18-5bcd-11eb-05b5-c72249e02975
begin
	fig_8_8 = plot(legend=:bottomright, xlabel="Computation time", ylabel="Value of start state under greedy policy")
	for b in [1, 3, 10]
		plot!(fig_8_8, mean(sweep(;b=b) for _ in 1:200), label="uniform b=$b")
		plot!(fig_8_8, mean(on_policy(;b=b) for _ in 1:200), label="on-policy b=$b")
	end
	fig_8_8
end

# ╔═╡ 8cba1ec8-5bcf-11eb-32c3-f38dee82139a
begin
	fig_8_8_2 = plot(legend=:bottomright, xlabel="Computation time", ylabel="Value of start state under greedy policy")
	plot!(fig_8_8_2, mean(sweep(;ns=10_000) for _ in 1:200), label="uniform")
	plot!(fig_8_8_2, mean(on_policy(;ns=10_000) for _ in 1:200), label="on-policy")
	fig_8_8_2
end


# ╔═╡ Cell order:
# ╟─48977b2e-5bc9-11eb-2ad7-958ca5fd2ecb
# ╠═4b149ad2-5bca-11eb-0104-0be8dd786177
# ╠═6bf4f104-5bca-11eb-2b4a-69c4ba7ffadc
# ╟─989f834a-5bca-11eb-363a-5f4afce32520
# ╠═97be6cb8-5bcb-11eb-3dc7-1b8105b206d7
# ╠═0c52737a-5bcd-11eb-1546-21885703f303
# ╠═aa061636-5bce-11eb-1381-bb00fb913f9b
# ╠═a466db90-5bcc-11eb-3385-e5117be33410
# ╠═9bf4aea2-5bcd-11eb-230f-cf63a009a0f9
# ╠═b92ba1e2-5bcb-11eb-16cf-f15929c2697b
# ╠═86538552-5bce-11eb-2cbf-ff6e175ef421
# ╠═f29c9f18-5bcd-11eb-05b5-c72249e02975
# ╠═8cba1ec8-5bcf-11eb-32c3-f38dee82139a
