using Ju
using Plots
gr()

figpath(f) = "docs/src/assets/figures/figure_$f.png"

const GridWorldLinearIndices = LinearIndices((4,4))
const GridWorldCartesianIndices = CartesianIndices((4,4))

isterminal(s::CartesianIndex{2}) = s == CartesianIndex(1,1) || s == CartesianIndex(4,4) 

function nextstep(s::CartesianIndex{2}, a::CartesianIndex{2})
    ns = s + a
    if isterminal(s) || ns[1] < 1 || ns[1] > 4 || ns[2] < 1 || ns[2] > 4
        ns = s
    end
    r = isterminal(s) ? 0. : -1.0
    [(nextstate=GridWorldLinearIndices[ns], reward=r, prob=1.0)]
end

const GridWorldActions = [CartesianIndex(-1, 0),
                          CartesianIndex(1,0),
                          CartesianIndex(0, 1),
                          CartesianIndex(0, -1)]

const GridWorldEnvModel = DeterministicDistributionModel([nextstep(GridWorldCartesianIndices[s], a) for s in 1:16, a in GridWorldActions])

function fig_4_1()
    V, π = TabularV(16), RandomPolicy(fill(0.25, 16, 4))
    policy_evaluation!(V, π, GridWorldEnvModel; γ=1.0)
    p = heatmap(1:4, 1:4, reshape(V.table, 4,4), yflip=true)
    savefig(p, figpath("4_1"))
    p
end