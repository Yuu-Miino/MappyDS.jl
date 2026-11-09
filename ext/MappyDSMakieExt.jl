module MappyDSMakieExt

using MappyDS: MappyDS, AbstractMap, PoincareMap, Trajectory, TrajectorySegment, PlotConfig, LineStyle, MappyError, as_state, dim, end_mode, image_detail
using Makie: Axis, Consume, Figure, Keyboard, Mouse, events, lines!, mouseposition, on, scatter!

_style_for(styles::LineStyle, mode, default::LineStyle) = styles
function _style_for(styles::Dict{Symbol, LineStyle}, mode, default::LineStyle)
    mode !== nothing && haskey(styles, mode) && return styles[mode]
    return get(styles, :_default, default)
end

_color(style::LineStyle) = (style.color, style.alpha)

function MappyDS.plot_trajectory(traj::Trajectory; ax = nothing, xkey::Integer = 1, ykey::Integer = 2, styles = LineStyle(; color = :black, alpha = 1.0), linewidth::Real = 1.0)
    if ax === nothing
        fig = Figure()
        ax = Axis(fig[1, 1])
    end
    default = LineStyle(; color = :black, alpha = 1.0)
    for seg in traj
        dim(seg) < max(xkey, ykey) && continue
        style = _style_for(styles, seg.mode, default)
        lines!(ax, seg.y[xkey, :], seg.y[ykey, :]; linewidth = linewidth, color = _color(style))
    end
    return ax
end

mutable struct PlotState
    y::Vector{Float64}
    mode::Union{Nothing, Symbol}
    params::Any
    only_map::Bool
    param_index::Int
    clicked::Bool
    stalled::Bool
end

_set_param(p::NamedTuple, key::Symbol, value) = merge(p, NamedTuple{(key,)}((value,)))
function _set_param(p::AbstractDict, key::Symbol, value)
    q = copy(p)
    q[key] = value
    return q
end

function MappyDS.plot2d(map::AbstractMap, y0, p = nothing, config::PlotConfig = PlotConfig(); mode = nothing, frames = nothing, show::Bool = true)
    is_pmap = map isa PoincareMap
    start_mode = nothing
    if is_pmap
        start_mode = mode === nothing ? map.initial_mode : mode
    elseif mode !== nothing
        throw(ArgumentError("mode is only supported for PoincareMap"))
    end
    state = PlotState(as_state(y0), start_mode, p, config.only_map, config.param_index, true, false)
    xkey, ykey = config.xkey, config.ykey

    fig = Figure(; size = config.figsize)
    ax = Axis(fig[1, 1]; xlabel = config.xlabel, ylabel = config.ylabel, limits = (config.xlim..., config.ylim...))

    iterate() = is_pmap ? image_detail(map, state.y, state.params; jacobian = false, trajectory = true, mode = state.mode) :
        image_detail(map, state.y, state.params; jacobian = false, trajectory = true)

    function advance!()
        state.stalled && return nothing
        clicked = state.clicked
        state.clicked = false
        result = try
            iterate()
        catch err
            err isa Union{MappyError, ErrorException, ArgumentError, DomainError} || rethrow()
            @warn "iteration failed: $(sprint(showerror, err)); click to choose a new initial state"
            state.stalled = true
            return nothing
        end
        state.y = result.y
        is_pmap && (state.mode = end_mode(result))
        segments = result.trajectory === nothing ? TrajectorySegment[] : [s for s in result.trajectory if dim(s) >= max(xkey, ykey)]
        if clicked && !isempty(segments)
            first = segments[1].y
            scatter!(ax, [first[xkey, 1]], [first[ykey, 1]]; markersize = config.markersize, color = _color(config.click_style))
        end
        state.only_map || MappyDS.plot_trajectory(Trajectory(segments); ax, xkey, ykey, styles = config.trajectory_style, linewidth = config.linewidth)
        if !isempty(segments)
            last = segments[end]
            px, py = last.y[xkey, end], last.y[ykey, end]
            style = _style_for(config.point_style, last.mode, LineStyle(; color = :red, alpha = 1.0))
        elseif length(result.y) >= max(xkey, ykey)
            px, py = result.y[xkey], result.y[ykey]
            style = _style_for(config.point_style, state.mode, LineStyle(; color = :red, alpha = 1.0))
        else
            return nothing
        end
        scatter!(ax, [px], [py]; markersize = config.markersize, color = _color(style))
        return nothing
    end

    if frames !== nothing
        for _ in 1:frames
            advance!()
        end
        show && display(fig)
        return fig
    end

    on(events(ax.scene).mousebutton) do ev
        if ev.button == Mouse.left && ev.action == Mouse.press
            pos = mouseposition(ax)
            y = copy(state.y)
            if is_pmap && start_mode !== nothing
                # Restart from the section the user chose; its dimension is known.
                state.mode = start_mode
                d = MappyDS.dom_dim(map.modes[start_mode])
                length(y) == d || (y = zeros(d))
            end
            if length(y) == 1
                y[1] = config.scalar_axis == :x ? pos[1] : pos[2]
            else
                y[xkey] = pos[1]
                y[ykey] = pos[2]
            end
            @info "restarting from $y"
            state.y = y
            state.clicked = true
            state.stalled = false
        end
        return Consume(false)
    end

    on(events(fig).keyboardbutton) do ev
        ev.action == Keyboard.press || return Consume(false)
        if ev.key == Keyboard.space || ev.key == Keyboard.e
            empty!(ax)
        elseif ev.key == Keyboard.f
            state.only_map = !state.only_map
            empty!(ax)
        elseif ev.key == Keyboard.p
            isempty(config.param_keys) && return Consume(false)
            state.param_index = mod1(state.param_index + 1, length(config.param_keys))
            @info "adjustable parameter: $(config.param_keys[state.param_index])"
        elseif ev.key == Keyboard.up || ev.key == Keyboard.down
            isempty(config.param_keys) && return Consume(false)
            key = config.param_keys[mod1(state.param_index, length(config.param_keys))]
            step = config.param_step * (ev.key == Keyboard.down ? -1 : 1)
            if state.params === nothing || !haskey(state.params, key)
                @warn "parameter $(repr(key)) is not in params"
                return Consume(false)
            end
            state.params = _set_param(state.params, key, round(state.params[key] + step; digits = 10))
            @info "parameters: $(NamedTuple(k => state.params[k] for k in config.param_keys))"
        end
        return Consume(false)
    end

    on(events(fig).tick) do _
        advance!()
        return nothing
    end

    show && display(fig)
    return fig
end

end # module
