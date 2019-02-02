using Ju
using Plots
gr()



const GridWorldLinearIndices = LinearIndices((5,5))
const GridWorldCartesianIndices = CartesianIndices((5,5))

function nextstep(s::CartesianIndex{2}, a::CartesianIndex{2})
    if s == CartesianIndex(1, 2)
        ns, r = CartesianIndex(5, 2), 10.
    elseif s == CartesianIndex(1, 4)
        ns, r = CartesianIndex(3, 4), 5.
    else
        ns = s + a
        if 1 ≤ ns[1] ≤ 5 && 1 ≤ ns[2] ≤ 5
            ns, r = ns, 0.
        else
            ns, r = s, -1.
        end
    end
    [(nextstate=GridWorldLinearIndices[ns], reward=r, prob=1.0)]
end

const GridWorldActions = [CartesianIndex(-1, 0),
                          CartesianIndex(1,0),
                          CartesianIndex(0, 1),
                          CartesianIndex(0, -1)]

const GridWorldEnvModel = DeterministicDistributionModel([nextstep(GridWorldCartesianIndices[s], a) for s in 1:25, a in GridWorldActions])

function fig_3_2()
    V, π = TabularV(25), RandomPolicy(fill(0.25, 25, 4))
    policy_evaluation!(V, π, GridWorldEnvModel)
    p = heatmap(1:5, 1:5, reshape(V.table, 5,5), yflip=true)
    savefig(p, "figure_3_2.png")
    p
end

function fig_3_5()
    V, π = TabularV(25), DeterministicPolicy(rand(1:4, 25), 4)
    policy_iteration!(V, π, GridWorldEnvModel)
    p = heatmap(1:5, 1:5, reshape(V.table, 5,5), yflip=true)
    savefig(p, "figure_3_5.png")
    p
end