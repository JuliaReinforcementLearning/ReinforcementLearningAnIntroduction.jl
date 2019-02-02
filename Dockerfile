FROM julia:1.1

ADD . /RLIntro
WORKDIR /RLIntro
RUN ["julia", "-e", "using Pkg; Pkg.Registry.add(\"General\"); Pkg.Registry.add(RegistrySpec(url = \"https://github.com/Ju-jl/Registry.git\")); Pkg.add(\"Plots\"); Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate(); pkg\"precompile\""]
CMD ["julia"]