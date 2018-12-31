@reexport module Chapter06RandomWalk
export fig_6_2_a, fig_6_2_b, fig_6_2_c
include("randomwalk.jl")
end

@reexport module Chapter06WindyGridWorld
export fig_6_2_d
include("windy_grid_world.jl")
end

@reexport module Chapter06CliffWalking
export fig_6_3_a, fig_6_3_b
include("cliff_walking.jl")
end

@reexport module Chapter06MaximizationBias
export fig_6_5
include("maximization_bias.jl")
end