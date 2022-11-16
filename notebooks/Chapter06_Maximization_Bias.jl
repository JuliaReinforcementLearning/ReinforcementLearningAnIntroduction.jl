### A Pluto.jl notebook ###
# v0.16.4

using Markdown
using InteractiveUtils

# ╔═╡ 9edb35ce-4ea7-11eb-353f-53459c337880
begin
	using ReinforcementLearning
	using Statistics
	using Flux
	using Plots
end

# ╔═╡ 6ce0a200-4ea7-11eb-123c-73e3a260351c
md"""
# Chapter 6.7 Maximization Bias and Double Learning

In example 6.7, authors introduced a MDP problem to compare the different performance of **Q-Learning** and **Double-Q-Learning**. This environment is kind of special compared to the environments we have seen before. In the first step, only `LEFT` and `RIGHT` are allowed. In the second step, if the `LEFT` is chosen previously, then we have 10 valid actions. We call this kind of environment is of `FULL_ACTION_SET`.
"""

# ╔═╡ 172236ac-4ea8-11eb-1393-5bc3683bf6b0
begin
	"""
	states:
	1:A
	2:B
	3:terminal
	actions:
	1: left
	2: right
	"""
	Base.@kwdef mutable struct MaximizationBiasEnv <: AbstractEnv
		position::Int = 1
		reward::Float64 = 0.0
	end
	function (env::MaximizationBiasEnv)(a::Int)
		if env.position == 1
			if a == 1
				env.position = 2
				env.reward = 0.0
			else
				env.position = 3
				env.reward = 0.0
			end
		elseif env.position == 2
			env.position = 3
			env.reward = randn() - 0.1
		end
		nothing
	end
end

# ╔═╡ 19876a70-4ea8-11eb-2b70-9d48875f7ae7
RLBase.state_space(env::MaximizationBiasEnv) = Base.OneTo(3)

# ╔═╡ 247ada66-4ea8-11eb-3c0a-af131b73e9f3
RLBase.action_space(env::MaximizationBiasEnv) = Base.OneTo(10)

# ╔═╡ 26b9f6f2-4ea8-11eb-38f3-3f1822c4acd2
RLBase.ActionStyle(env::MaximizationBiasEnv) = FULL_ACTION_SET

# ╔═╡ 296e0ab6-4ea8-11eb-33c5-71be241708b1
const LEFT = 1

# ╔═╡ 2c06d25a-4ea8-11eb-26e3-01d81d714859
const RIGHT = 2

# ╔═╡ 2e11adfc-4ea8-11eb-02bc-d550994774f8
function RLBase.legal_action_space(env::MaximizationBiasEnv)
    if env.position == 1
        (LEFT, RIGHT)
    else
        Base.OneTo(10)
    end
end

# ╔═╡ 3148b560-4ea8-11eb-0fc1-f77efb282ba4
function RLBase.legal_action_space_mask(env::MaximizationBiasEnv)
    m = fill(false, 10)
    if env.position == 1
        m[LEFT] = true
        m[RIGHT] = true
    else
        m .= true
    end
    m
end

# ╔═╡ 3a674788-4ea8-11eb-1931-b39142fffb47
function RLBase.reset!(env::MaximizationBiasEnv)
    env.position = 1
    env.reward = 0.0
    nothing
end

# ╔═╡ 3ec32b6c-4ea8-11eb-1575-319c528b4f02
RLBase.reward(env::MaximizationBiasEnv) = env.reward

# ╔═╡ 4127acde-4ea8-11eb-0e47-7d82bb5cb310
RLBase.is_terminated(env::MaximizationBiasEnv) = env.position == 3

# ╔═╡ 4309b632-4ea8-11eb-1d46-edd93efd6925
RLBase.state(env::MaximizationBiasEnv) = env.position

# ╔═╡ 45650d82-4ea8-11eb-39c2-71fc325d6c13
md"""
Now the environment is well defined.
"""

# ╔═╡ 56f8ba1c-4ea8-11eb-2ce7-8357675ee6ef
world = MaximizationBiasEnv()

# ╔═╡ 657eca7c-4ea8-11eb-3c42-b1a3085f6129
NS, NA = length(state_space(world)), length(action_space(world))

# ╔═╡ 8d15e872-4ea8-11eb-2fc2-7f3c4a90a85a
md"""
To calculate the percentage of chosing `LEFT` action in the first step, we'll create customized hook here:
"""

# ╔═╡ 6f798d14-4ea8-11eb-2825-8f6f17ff4f05
begin
	Base.@kwdef mutable struct CountOfLeft <: AbstractHook
		counts::Vector{Bool} = []
	end
	function (f::CountOfLeft)(::PreActStage, agent, env, action)
		if state(env) == 1
			push!(f.counts, action == LEFT)
		end
	end
end

# ╔═╡ d026cbb8-4ea8-11eb-29a2-dd75efe64466
md"""
Next we create two agent factories, Q-Learning and Double-Q-Learning.
"""

# ╔═╡ ea42c3c6-4ea8-11eb-1d82-0566785eeef3
create_double_Q_agent() = Agent(
    policy=QBasedPolicy(
        learner=DoubleLearner(
            L1=TDLearner(
                approximator=TabularQApproximator(
                    n_state=NS,
                    n_action=NA,
                    opt=Descent(0.1),
                ),
                method=:SARS,
                γ=1.,
                n=0,
            ),
            L2=TDLearner(
                approximator=TabularQApproximator(;
                    n_state=NS,
                    n_action=NA,
                    opt=Descent(0.1),
                ),
                method=:SARS,
                γ=1.,
                n=0,
            ),
            ),
        explorer=EpsilonGreedyExplorer(0.1;is_break_tie=true)
    ),
    trajectory=VectorSARTTrajectory()
)

# ╔═╡ eccb09aa-4ea8-11eb-08a0-2f6344c38669
create_Q_agent() = Agent(
    policy=QBasedPolicy(
        learner=TDLearner(
            approximator=TabularQApproximator(;n_state=NS, n_action=NA, opt=Descent(0.1)),
            method=:SARS,
            γ=1.0,
            n=0
            ),
        explorer=EpsilonGreedyExplorer(0.1;is_break_tie=true)
        ),
    trajectory=VectorSARTTrajectory()
)

# ╔═╡ 0e4b6994-4ea9-11eb-111f-7b37d6a716e5
begin
	DQ_stats = []
	for _ in 1:1000
		hook = CountOfLeft()
		run(create_double_Q_agent(), world, StopAfterEpisode(300),hook)
		push!(DQ_stats, hook.counts)
	end
	plot(mean(DQ_stats)*100, legend=:topright, label="double q", xlabel="Episodes", ylabel="% left actions from A")
	
	Q_stats = []
	for _ in 1:1000
		hook = CountOfLeft()
		run(create_Q_agent(), world, StopAfterEpisode(300),hook)
		push!(Q_stats, hook.counts)
	end
	plot!(mean(Q_stats)*100, legend=:topright, label="q")
	hline!([5], linestyle=:dash, label="optimal")
end


# ╔═╡ Cell order:
# ╟─6ce0a200-4ea7-11eb-123c-73e3a260351c
# ╠═9edb35ce-4ea7-11eb-353f-53459c337880
# ╠═172236ac-4ea8-11eb-1393-5bc3683bf6b0
# ╠═19876a70-4ea8-11eb-2b70-9d48875f7ae7
# ╠═247ada66-4ea8-11eb-3c0a-af131b73e9f3
# ╠═26b9f6f2-4ea8-11eb-38f3-3f1822c4acd2
# ╠═296e0ab6-4ea8-11eb-33c5-71be241708b1
# ╠═2c06d25a-4ea8-11eb-26e3-01d81d714859
# ╠═2e11adfc-4ea8-11eb-02bc-d550994774f8
# ╠═3148b560-4ea8-11eb-0fc1-f77efb282ba4
# ╠═3a674788-4ea8-11eb-1931-b39142fffb47
# ╠═3ec32b6c-4ea8-11eb-1575-319c528b4f02
# ╠═4127acde-4ea8-11eb-0e47-7d82bb5cb310
# ╠═4309b632-4ea8-11eb-1d46-edd93efd6925
# ╟─45650d82-4ea8-11eb-39c2-71fc325d6c13
# ╠═56f8ba1c-4ea8-11eb-2ce7-8357675ee6ef
# ╠═657eca7c-4ea8-11eb-3c42-b1a3085f6129
# ╟─8d15e872-4ea8-11eb-2fc2-7f3c4a90a85a
# ╠═6f798d14-4ea8-11eb-2825-8f6f17ff4f05
# ╟─d026cbb8-4ea8-11eb-29a2-dd75efe64466
# ╠═ea42c3c6-4ea8-11eb-1d82-0566785eeef3
# ╠═eccb09aa-4ea8-11eb-08a0-2f6344c38669
# ╠═0e4b6994-4ea9-11eb-111f-7b37d6a716e5
