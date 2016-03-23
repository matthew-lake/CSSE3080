ENV["MOCHA_USE_CUDA"] = "true"

using Mocha
using JLD
srand(12345678)

dataLayer = AsyncHDF5DataLayer(name="train-data", source="train.txt", batch_size=64, shuffle=false)

shallowLayers = [
	ConvolutionLayer(name="conv1", n_filter=20, kernel=(5,5), bottoms=[:data], tops=[:conv])
	PoolingLayer(name="pool1", kernel=(2,2), stride=(2,2), bottoms=[:conv], tops=[:pool])
	InnerProductLayer(name="ip1", output_dim=18, bottoms=[:pool], tops=[:ip1])
	SoftmaxLossLayer(name="loss", bottoms=[:ip1,:label])
]

backend = DefaultBackend()
init(backend)

net = Net("MNIST-train", backend, [dataLayer, shallowLayers...])
println(net)
open("net.dot", "w") do out net2dot(out, net) end

f = jldopen("shallowNet.jld", "r")
load_network(f, net)

array = Array{Float32, 2}(size(net.output_blobs[:ip1])...)
@show net.output_blobs[:ip1]
copy!(net.output_blobs[:ip1], array)
@show array

exp_dir = "snapshots-$(Mocha.default_backend_type)"

method = SGD()
params = make_solver_parameters(method, max_iter=100, regu_coef=0.0005,
                                mom_policy=MomPolicy.Fixed(0.9),
                                lr_policy=LRPolicy.Inv(0.01, 0.0001, 0.75),
                                load_from=exp_dir)
solver = Solver(method, params)

setup_coffee_lounge(solver, save_into="$exp_dir/statistics.jld", every_n_iter=1000)

# report training progress every 100 iterations
add_coffee_break(solver, TrainingSummary(), every_n_iter=100)

# save snapshots every 5000 iterations
#add_coffee_break(solver, Snapshot(exp_dir), every_n_iter=5000)

# show performance on test data every 1000 iterations
data_layer_test = HDF5DataLayer(name="test-data", source="test.txt", batch_size=100)
acc_layer = SquareLossLayer(name="test-accuracy", bottoms=[:ip1, :label])
test_net = Net("MNIST-test", backend, [data_layer_test, shallowLayers...])
add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=1000)

solve(solver, net)

@show net.output_blobs[:ip1]

f = jldopen("shallowNet.jld", "w")
save_network(f, net)
close(f)

#Profile.init(int(1e8), 0.001)
#@profile solve(solver, net)
#open("profile.txt", "w") do out
#  Profile.print(out)
#end

destroy(net)
destroy(test_net)
shutdown(backend)
