# Containers for computed trajectories.

"""
    TrajectorySegment(mode, kind, y, t = nothing)

The part of a trajectory spent in a single mode. `kind` is `:continuous` or
`:discrete`, `y` holds the sampled states as columns (`dim × n_points`; a
discrete segment holds the two points before and after the map) and `t` the
sampling times relative to the start of the segment (`nothing` for discrete
segments).
"""
struct TrajectorySegment
    mode::Symbol
    kind::Symbol
    y::Matrix{Float64}
    t::Union{Nothing, Vector{Float64}}

    function TrajectorySegment(mode::Symbol, kind::Symbol, y::AbstractMatrix, t = nothing)
        kind in (:continuous, :discrete) || throw(ArgumentError("kind must be :continuous or :discrete, got $(repr(kind))"))
        tt = t === nothing ? nothing : Vector{Float64}(t)
        tt === nothing || length(tt) == size(y, 2) || throw(DimensionMismatch("t has $(length(tt)) entries for $(size(y, 2)) states"))
        return new(mode, kind, Matrix{Float64}(y), tt)
    end
end

"""
    dim(x)

Dimension of the state space of a trajectory segment or of a map.
"""
dim(seg::TrajectorySegment) = size(seg.y, 1)

"""
    Trajectory(segments = TrajectorySegment[])

A trajectory as a sequence of [`TrajectorySegment`](@ref)s in chronological
order. Modes with `in_trajectory = false` do not contribute a segment.

`traj.y` concatenates the states of all segments (`dim × n_points`); it
throws an `ArgumentError` if the segments have different dimensions.
"""
struct Trajectory
    segments::Vector{TrajectorySegment}
end
Trajectory() = Trajectory(TrajectorySegment[])

Base.length(traj::Trajectory) = length(traj.segments)
Base.iterate(traj::Trajectory, state...) = iterate(traj.segments, state...)
Base.eltype(::Type{Trajectory}) = TrajectorySegment
Base.getindex(traj::Trajectory, i) = traj.segments[i]
Base.lastindex(traj::Trajectory) = lastindex(traj.segments)
Base.propertynames(::Trajectory) = (:segments, :y)

function Base.getproperty(traj::Trajectory, name::Symbol)
    name === :y && return states(traj)
    return getfield(traj, name)
end

"""
    states(traj::Trajectory) -> Matrix{Float64}

All sampled states of the trajectory concatenated, `dim × n_points` (also
available as `traj.y`).
"""
function states(traj::Trajectory)
    segs = getfield(traj, :segments)
    isempty(segs) && return zeros(0, 0)
    dims = unique(dim(s) for s in segs)
    length(dims) == 1 || throw(ArgumentError("segments have different dimensions: $(sort(dims))"))
    return reduce(hcat, (s.y for s in segs))
end
