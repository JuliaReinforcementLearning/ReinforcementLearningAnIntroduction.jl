using Ju
using Distributions
using Plots
gr()

figpath(f) = "docs/src/assets/figures/figure_$f.png"

const PoissonUpperBound = 10
const MaxCars= 20
const MaxMoves = 5
const CostPerMove = 2
const CarRentalCartesianIndices = CartesianIndices((0:MaxCars,0:MaxCars))
const CarRentalLinearIndices = LinearIndices((0:MaxCars,0:MaxCars))
const Actions = -MaxMoves:MaxMoves
const RequestDist_1 = Poisson(3)
const RequestDist_2 = Poisson(4)
const ReturnDist_1 = Poisson(3)
const ReturnDist_2 = Poisson(2)

decode_state(s::Int) = Tuple(CarRentalCartesianIndices[s])
encode_state(s1::Int, s2::Int) = CarRentalLinearIndices[CartesianIndex(s1+1, s2+1)]
decode_action(a::Int) = a - MaxMoves - 1
encode_action(a::Int) = a + MaxMoves + 1

"reduce calculation"
function merge_prob(dist)
    merged = Dict()
    for (s′, r, p) in dist
        if haskey(merged, (s′, r))
            merged[(s′, r)] += p
        else
            merged[(s′, r)] = p
        end
    end
    [(nextstate=s′, reward=r, prob=p) for ((s′, r), p) in merged]
end

function nextstep(state::Int, action::Int)
    (s1, s2), a = decode_state(state), decode_action(action)
    move = a > 0 ? min(a, s1) : max(a, -s2)
    reward = -CostPerMove*abs(move)
    s1′, s2′ = min(s1 - move, MaxCars), min(s2 + move, MaxCars)
    merge_prob((nextstate = encode_state(min(max(s1′-req_1, 0)+ret_1, MaxCars), min(max(s2′-req_2, 0)+ret_2, MaxCars)),  # cars returned today can only be used in the next day
                reward    = reward + (min(s1′, req_1) + min(s2′, req_2)) * 10,
                prob      = pdf(RequestDist_1, req_1) * pdf(RequestDist_2, req_2) * pdf(ReturnDist_1, ret_1) * pdf(ReturnDist_2, ret_2))
                for req_1 in 0:PoissonUpperBound, req_2 in 0:PoissonUpperBound, ret_1 in 0:PoissonUpperBound, ret_2 in 0:PoissonUpperBound)
end

const CarRentalEnvModel = DeterministicDistributionModel([nextstep(s, a) for s in 1:(MaxCars+1)^2, a in 1:length(Actions)])

function fig_4_2(max_iter=100)
    V, π = TabularV((1+MaxCars)^2), DeterministicPolicy(zeros(Int,21^2), length(Actions))
    policy_iteration!(V, π, CarRentalEnvModel; γ=0.9, max_iter=max_iter)
    p1 = heatmap(0:MaxCars, 0:MaxCars, reshape([decode_action(x) for x in π.table], 1+MaxCars,1+MaxCars))
    savefig(p1, figpath("4_2_policy"))
    p2 = heatmap(0:MaxCars, 0:MaxCars, reshape(V.table, 1+MaxCars,1+MaxCars))
    savefig(p2, figpath("4_2_value"))
    p1, p2
end