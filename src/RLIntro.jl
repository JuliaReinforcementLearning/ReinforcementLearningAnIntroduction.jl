module RLIntro

export plot_all

include("environments/environments.jl")

using Reexport
include("chapter01/chapter01.jl")
include("chapter02/chapter02.jl")
include("chapter03/chapter03.jl")
include("chapter04/chapter04.jl")
include("chapter05/chapter05.jl")
include("chapter06/chapter06.jl")
include("chapter07/chapter07.jl")
include("chapter08/chapter08.jl")
include("chapter09/chapter09.jl")
include("chapter10/chapter10.jl")
include("chapter11/chapter11.jl")
include("chapter12/chapter12.jl")
include("chapter13/chapter13.jl")

function plot_all(fig_dir=".")
    for f in names(RLIntro)
        if startswith(string(f), "fig")
            @eval $f()
        end
    end
end

end # module
