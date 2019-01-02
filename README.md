<div align="center"> 
<a href="http://incompleteideas.net/book/the-book-2nd.html">
<img src="https://tianjun.me/static/site_resources/img/RLIntro2Cover-min.jpg" alt="RLIntro2Cover-min.jpg" title="RLIntro" width="200"/> 
</a>
<blockquote> 
<p> "To think is to forget a difference, to generalize, to abstract."</p>
<p>― <a href="https://en.wikipedia.org/wiki/Jorge_Luis_Borges">Jorge Luis Borges</a>, <a href="https://en.wikipedia.org/wiki/Funes_the_Memorious">Funes the Memorious</a></p>
</blockquote>
</div>

<hr>

This repo reproduces almost all the figures on the book [Reinforcement Learning: An Introduction(2nd)](http://incompleteideas.net/book/the-book-2nd.html).

# Workflow

## Reproduce

Just run the following command to install this package:

```bash
$ julia -e "using Pkg; Pkg.add(\"Plots\"); Pkg.add(PackageSpec(url=\"https://github.com/Ju-jl/Ju.jl.git\")); Pkg.add(PackageSpec(url=\"https://github.com/Ju-jl/ReinforcementLearningAnIntroduction.jl\"));"
```

Then enter the REPL:

```julia
julia> using RLIntro  # Hold on! It might take several minutes to pre-compile

julia> @show [f for f in names(RLIntro) if startswith(string(f), "fig")];  # list all the functions to reproduce corresponding figures
[f for f = names(RLIntro) if startswith(string(f), "fig")] = Symbol[:fig_10_1, :fig_10_2, :fig_10_3, :fig_10_4, :fig_10_5, :fig_11_2, :fig_12_3, :fig_13_1, :fig_13_2, :fig_2_1, :fig_2_2, :fig_2_3, :fig_2_4, :fig_2_5, :fig_2_6, :fig_3_2, :fig_3_5, :fig_4_1, :fig_4_2, :fig_4_3, :fig_5_1, :fig_5_2, :fig_5_3, :fig_5_4, :fig_6_2_a, :fig_6_2_b, :fig_6_2_c, :fig_6_2_d, :fig_6_3_a, :fig_6_3_b, :fig_6_5, :fig_7_2, :fig_8_2, :fig_8_4, :fig_8_4_example, :fig_8_5, :fig_8_7, :fig_8_8, :fig_9_1, :fig_9_10, :fig_9_2_a, :fig_9_2_b, :fig_9_5]

julia> fig_2_2()  # reproduce figure_2_2
```

**Notice** that for some figures you may need to install *pdflatex*.

## Develop

If you would like to make some improvements, I'd suggest the following workflow:

1. Clone this repo and enter the project folder.
1. Enter the pkg mode and `(RLIntro) pkg> add https://github.com/Ju-jl/Ju.jl.git` (Because the `Ju.j` is not registered yet. It will not be a big problem after Julia 1.1 get released)
1. Make changes to some existing *Environment* or create a new Environment and include it in the REPL (like `include("src/environments/MultiArmBandits.jl")`)
1. Make changes to the related source codes and include it in the REPL (like `include("src/chapter02/ten_armed_testbed.jl")`)
1. Run the functions to draw figures (`fig_2_2()`).
1. Repeat the above three steps.

# Contents

| Chapters | Figures | Description |
|---|:--      | :-- |
| Chapter01 |  | Run `play()` will prompt an interactive interface to play the [TicTacToe](https://en.wikipedia.org/wiki/Tic-tac-toe) game. |
| Chapter02 | [fig_2_2](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_2_2.png), [fig_2_3](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_2_3.png), [fig_2_4](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_2_4.png), [fig_2_5](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_2_5.png), [fig_2_6](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_2_6.png) | |
| Chapter03 | [fig_3_2](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_3_2.png), [fig_3_5](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_3_5.png)| Here the heatmap is used to represent the value|
| Chapter04 | [fig_4_1](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_4_1.png), [fig_4_2_policy](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_4_2_policy.png), [fig_4_2_value](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_4_2_value.png), [fig_4_3](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_4_3.png)| |
| Chapter05 | [fig_5_1_no_usable_ace_n_10000](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_5_1_no_usable_ace_n_10000.png), [fig_5_1_usable_ace_n_10000](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_5_1_usable_ace_n_10000.png) | Warning!!! The result is different to the figures on the book. Please help to correct it.|
| | [fig_5_2_no_usable_ace_n_500000](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_5_2_no_usable_ace_n_500000.png), [fig_5_2_usable_ace_n_500000](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_5_2_usable_ace_n_500000.png), [fig_5_2_no_usable_ace_policy_n_500000](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_5_2_no_usable_ace_policy_n_500000.png), [fig_5_2_usable_ace_policy_n_500000](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_5_2_usable_ace_policy_n_500000.png) | |
| | [fig_5_3](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_5_3.png), [fig_5_4](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_5_4.png)| |
| Chapter06 | [fig_6_2_a](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_6_2_a.png), [fig_6_2_b](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_6_2_b.png), [fig_6_2_c](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_6_2_c.png), [fig_6_2_d](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_6_2_d.png) | |
| | [fig_6_3_a](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_6_3_a.png), [fig_6_3_b](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_6_3_b.png) | |
| | [fig_6_5](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_6_5.png) | |
| Chapter07 | [fig_7_2](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_7_2.png) | |
| Chapter08 | [fig_8_2](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_8_2.png), [fig_8_4_example](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_8_4_example.png), [fig_8_4](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_8_4.png) | |
| | [fig_8_5](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_8_5.png), [fig_8_7](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_8_7.png), [fig_8_8_a](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_8_8_a.png), [fig_8_8_b](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_8_8_b.png) | |
| Chapter09 | [fig_9_1_a](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_9_1_a.png), [fig_9_1_b](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_9_1_b.png), [fig_9_2_a](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_9_2_a.png), [fig_9_2_b](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_9_2_b.png), [fig_9_5](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_9_5.png)| |
| Chapter10 | [fig_10_1_1](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_10_1_1.png), [fig_10_1_2](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_10_1_2.png), [fig_10_1_3](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_10_1_3.png), [fig_10_1_4](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_10_1_4.png), [fig_10_1_5](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_10_1_5.png) | |
| | [fig_10_3](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_10_3.png), [fig_10_4](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_10_4.png), [fig_10_5](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_10_5.png)| |
|Chapter11 | [fig_11_2](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_11_2.png) | |
|Chapter12 | [fig_12_3](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_12_3.png)| Other figures in Chapter12 are not that easy to reproduce by using the Ju.jl package. You may take a try and correct me with a PR.|
| Chapter13 | [fig_13_1](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_13_1.png), [fig_13_2](https://raw.githubusercontent.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/master/docs/src/assets/figures/figure_13_2.png) | Figure_13_2 is a slightly different to the original figure on the book.|

# Related Packages

- [Ju.jl](https://github.com/Ju-jl/Ju.jl)

    This repo mainly relies on [Ju.jl](https://github.com/Ju-jl/Ju.jl)
- [ShangtongZhang/reinforcement-learning-an-introduction](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)

    You may also take a look at the Python code.