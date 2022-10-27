### A Pluto.jl notebook ###
# v0.16.4

using Markdown
using InteractiveUtils

# ╔═╡ 970efd92-5c85-11eb-180e-d508223139fe
begin
	using ReinforcementLearning
	using Flux
	using Statistics
	using Plots
end

# ╔═╡ e939f502-5c8c-11eb-3e40-1386be4610c3
begin
	using Distributions


	const N_SERVERS = 10
	const PRIORITIES = [1.0, 2.0, 4.0, 8.0]
	const FREE_PROB = 0.06
	const ACCEPT_REJECT = (:accept, :reject)
	const CUSTOMERS = 1:length(PRIORITIES)

	const TRANSFORMER = LinearIndices((0:N_SERVERS, CUSTOMERS))

	Base.@kwdef mutable struct AccessControlEnv <: AbstractEnv
		n_servers::Int = 10
		n_free_servers::Int = 0
		customer::Int = rand(CUSTOMERS)
		reward::Float64 = 0.0
	end

	RLBase.state_space(env::AccessControlEnv) = Base.OneTo(length(TRANSFORMER))
	RLBase.action_space(env::AccessControlEnv) = Base.OneTo(2)

	function (env::AccessControlEnv)(a)
		action, reward = ACCEPT_REJECT[a], 0.0
		if env.n_free_servers > 0 && action == :accept
			env.n_free_servers -= 1
			reward = PRIORITIES[env.customer]
		end

		env.n_free_servers += rand(Binomial(env.n_servers - env.n_free_servers, FREE_PROB))
		env.customer = rand(CUSTOMERS)
		env.reward = reward

		nothing
	end

	RLBase.reward(env::AccessControlEnv) = env.reward
	RLBase.is_terminated(env::AccessControlEnv) = false
	RLBase.state(env::AccessControlEnv) = TRANSFORMER[CartesianIndex(env.n_free_servers + 1, env.customer)]

	function RLBase.reset!(env::AccessControlEnv)
		env.n_free_servers = env.n_servers
		env.customer = rand(CUSTOMERS)
		env.reward = 0.0
		nothing
	end
end

# ╔═╡ 01d9380a-5c85-11eb-1463-b1373c6cfe72
md"""
# Chapter 10.3 Access Control

In Chapter 10.3, an algorithm of **Differential semi-gradient Sarsa for estimating q̂ ≈ q⋆**. This one is not included in `ReinforcementLearning.jl`. Here we'll use it as an example to demonstrate how easy it is to extend components in `ReinforcementLearning.jl`.
"""

# ╔═╡ 40f4eae6-5c87-11eb-1978-c9efd63427e9
md"""
## Implement the DifferentialTDLearner

First let's define a `DifferentialTDLearner`. It will be used to estimate Q values. So we have to implement `(L::DifferentialTDLearner)(env::AbstractEnv)`, which simply forward the current `state` to the inner `approximator`.
"""

# ╔═╡ 30eb366e-5c87-11eb-12ad-09087fc4d788
begin
	Base.@kwdef mutable struct DifferentialTDLearner{A<:AbstractApproximator} <: AbstractLearner
		approximator::A
		β::Float64
		R̄::Float64 = 0.0
	end
	(L::DifferentialTDLearner)(env::AbstractEnv) = L.approximator(state(env))
end

# ╔═╡ 9c550740-5c87-11eb-11f1-63afc18d1229
md"""
Now based on the definition of this algorithm on the book， we can implement the updating logic as follows:
"""

# ╔═╡ a223544c-5c87-11eb-1728-8f83730232e7
function RLBase.update!(L::DifferentialTDLearner, transition::Tuple)
    s, a, r, t, s′, a′ = transition
    β, Q = L.β, L.approximator
    δ = r - L.R̄ + (1-t)*Q(s′, a′) - Q(s,a)
    L.R̄ += β* δ
    update!(Q, (s,a) => -δ)
end

# ╔═╡ 17a1328e-5c88-11eb-3603-01c70fd05502
md"""
Next, we dispatch some runtime logic to our specific learner to make sure the above `update!` function is called at the right time.
"""

# ╔═╡ 3dfbe3b6-5c88-11eb-291b-45a716887e3c
begin
	function RLBase.update!(
		p::DifferentialTDLearner,
		t::AbstractTrajectory,
		e::AbstractEnv,
		s::PreActStage,
	)
		if length(t[:terminal]) > 0
			update!(
				p,
				(
					t[:state][end-1],
					t[:action][end-1],
					t[:reward][end],
					t[:terminal][end],
					t[:state][end],
					t[:action][end]
				)
			)
		end
	end

	function RLBase.update!(
		p::DifferentialTDLearner,
		t::AbstractTrajectory,
		e::AbstractEnv,
		s::PostEpisodeStage,
	)
		update!(
			p,
			(
				t[:state][end-1],
				t[:action][end-1],
				t[:reward][end],
				t[:terminal][end],
				t[:state][end],
				t[:action][end]
			)
		)
	end
end

# ╔═╡ 435d632a-5c88-11eb-314b-c1c580f35b70
md"""
The above function specifies that we only update the `DifferentialTDLearnerTDLearner` at the `PreActStage` and the `PostEpisodeStage`. Also note that we don't need to push all the transitions into the trajectory at each step. So we can `empty!` it at the start of an episode:
"""

# ╔═╡ 171efdc4-5c8c-11eb-3aa1-d5904fa31fbe
function RLBase.update!(
    t::AbstractTrajectory,
    ::QBasedPolicy{<:DifferentialTDLearner},
    ::AbstractEnv,
    ::PreEpisodeStage
)
    empty!(t)
end

# ╔═╡ 35efbb7e-5c8c-11eb-36b0-3ffc1355bf59
md"""
## Access Control Environment

Before evaluating the learner implemented above, we have to first define the environment.
"""

# ╔═╡ ffa3b58a-5c8c-11eb-3339-cbd7f96a0164
world = AccessControlEnv()

# ╔═╡ 178e097a-5c8d-11eb-24f2-f3a64cfa7bd8
NS = length(state_space(world))

# ╔═╡ 1ef2ece2-5c8d-11eb-22ce-c746cd951bf2
NA = length(action_space(world))

# ╔═╡ 144c10fe-5c8d-11eb-222f-95c694c1cae6
agent = Agent(
    policy=QBasedPolicy(
        learner=DifferentialTDLearner(
            approximator=TabularQApproximator(
				;n_state=NS,
				n_action=NA,
				opt=Descent(0.01)
				),
            β=0.01,
        ),
        explorer=EpsilonGreedyExplorer(0.1)
    ),
    trajectory=VectorSARTTrajectory()
)

# ╔═╡ 32d8b7a2-5c8d-11eb-15c7-d72a637c7374
run(agent, world, StopAfterStep(2*10^6; is_show_progress=false))

# ╔═╡ 3bdff022-5c8d-11eb-37ff-355f604d3f56
begin

	p = plot(legend=:bottomright, xlabel="Number of free servers", ylabel="Differential value of best action")
	for i in 1:length(PRIORITIES)
		plot!(
			[agent.policy.learner.approximator(TRANSFORMER[(CartesianIndex(n+1, i))]) |> maximum
				for n in 1:N_SERVERS],
			label="priority = $(PRIORITIES[i])")
	end
	p
end

# ╔═╡ Cell order:
# ╟─01d9380a-5c85-11eb-1463-b1373c6cfe72
# ╠═970efd92-5c85-11eb-180e-d508223139fe
# ╟─40f4eae6-5c87-11eb-1978-c9efd63427e9
# ╠═30eb366e-5c87-11eb-12ad-09087fc4d788
# ╟─9c550740-5c87-11eb-11f1-63afc18d1229
# ╠═a223544c-5c87-11eb-1728-8f83730232e7
# ╟─17a1328e-5c88-11eb-3603-01c70fd05502
# ╠═3dfbe3b6-5c88-11eb-291b-45a716887e3c
# ╟─435d632a-5c88-11eb-314b-c1c580f35b70
# ╠═171efdc4-5c8c-11eb-3aa1-d5904fa31fbe
# ╟─35efbb7e-5c8c-11eb-36b0-3ffc1355bf59
# ╠═e939f502-5c8c-11eb-3e40-1386be4610c3
# ╠═ffa3b58a-5c8c-11eb-3339-cbd7f96a0164
# ╠═178e097a-5c8d-11eb-24f2-f3a64cfa7bd8
# ╠═1ef2ece2-5c8d-11eb-22ce-c746cd951bf2
# ╠═144c10fe-5c8d-11eb-222f-95c694c1cae6
# ╠═32d8b7a2-5c8d-11eb-15c7-d72a637c7374
# ╠═3bdff022-5c8d-11eb-37ff-355f604d3f56
