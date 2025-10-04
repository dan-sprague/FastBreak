"""
    write_stan_data(filename, x, y, n_breakpoints)

Write data to JSON file for fastbreak.stan

# Arguments
- `filename::String`: output filename (e.g., "data.json")
- `x::Vector`: predictor variable
- `y::Vector`: response variable
- `n_breakpoints::Int`: number of breakpoints

# Example
```julia
x = collect(0.0:0.1:10.0)
y = sin.(x) + randn(length(x)) * 0.1
write_stan_data("mydata.json", x, y, 4)
```
"""
function write_stan_data(filename::String, x::Vector, y::Vector, n_breakpoints::Int)
    data = Dict(
        "N" => length(x),
        "K" => n_breakpoints,
        "x" => x,
        "y" => y,
    )

    open(filename, "w") do f
        JSON.print(f, data, 2)  # 2 = indentation level
    end

    println("Wrote Stan data to $filename")
    println("  N = $(data["N"]) observations")
    println("  K = $(data["K"]) breakpoints")
end

"""
    write_stan_data(filename, model::SegmentedModel)

Write data to JSON file for fastbreak.stan from a SegmentedModel

# Example
```julia
model = SegmentedModel(x, y, 4)
write_stan_data("mydata.json", model)
```
"""
function write_stan_data(filename::String, model::SegmentedModel)
    write_stan_data(filename, model.x, model.y, model.n_breakpoints)
end
