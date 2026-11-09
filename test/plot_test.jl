@testitem "plotting through the Makie extension" setup = [Fixtures] begin
    using CairoMakie
    hmap = henon_map()
    traj = trajectory(hmap, [0.1, 0.1], HENON_PARAMS; n = 5)
    ax = plot_trajectory(traj)
    @test ax isa Makie.Axis
    @test length(ax.scene.plots) == 5
    fig = Figure()
    ax2 = Axis(fig[1, 1])
    @test plot_trajectory(traj; ax = ax2, styles = Dict(:map => LineStyle(; color = :teal, alpha = 0.5))) === ax2
    @test length(ax2.scene.plots) == 5
    # Segments that do not contain the plotted components are skipped.
    izh = make_izhikevich_map()
    low = trajectory(izh, -3.0, IZH_PARAMS; n = 2)
    ax3 = plot_trajectory(low; xkey = 1, ykey = 3)
    @test length(ax3.scene.plots) == 0

    config = PlotConfig(; xlim = (-2.0, 2.0), ylim = (-1.0, 1.0), param_keys = [:a, :b])
    fig2 = plot2d(hmap, [0.1, 0.1], HENON_PARAMS, config; frames = 10, show = false)
    @test fig2 isa Makie.Figure
    axes = [c for c in fig2.content if c isa Makie.Axis]
    @test length(axes) == 1
    # 10 image points, 10 two-point trajectory segments and the clicked start.
    @test length(axes[1].scene.plots) == 21
    only = plot2d(hmap, [0.1, 0.1], HENON_PARAMS, PlotConfig(; only_map = true); frames = 3, show = false)
    @test length([c for c in only.content if c isa Makie.Axis][1].scene.plots) == 4

    pfig = plot2d(izh, -3.0, IZH_PARAMS, PlotConfig(; xlim = (-60.0, 40.0), ylim = (-10.0, 5.0)); frames = 3, show = false)
    @test pfig isa Makie.Figure
    @test_throws ArgumentError plot2d(hmap, [0.1, 0.1], HENON_PARAMS; mode = :map, frames = 1, show = false)
end
