FROM julia:1.1

ADD . /RLIntro
WORKDIR /RLIntro
RUN ["julia", "-e", "using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate(); pkg\"precompile\""]
CMD ["julia"]