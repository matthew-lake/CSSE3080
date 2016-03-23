using Optim
using Devectorize

"""
Linear Regression Cost function with Regularisation

Parameters:
	params      = vector containing rolled up X and Θ
		X       = array of features for each movie
		Θ       = array of weights for each user
	Y           = array of ratings for each movie
	R           = array of the existance of ratings for each movie
	numUsers    = number of users
	numMovies   = number of movies
	numFeatures = number of features
	λ           = regularisation constant

Returns:
	J = linear regression cost of the given weights with the given regularisation constant
"""
function cost(params, Y::Array{Int}, numUsers, numMovies, numFeatures, λ)
	# Unfold params -> X, Θ
	X, Θ = unroll(params, numUsers, numMovies, numFeatures)

	# Calculate cost
	J = 0
 	@simd for i = 1:size(Y)[2]
		@inbounds J += (dot(X[Y[2, i], :][:], Θ[Y[1, i], :][:]) - Y[3, i])^2
	end
	J = 1/2 * J
	# add regularisation
	J += λ/2 * (sum(Θ.^2) + sum(X.^2))

	return J
end

"""
Linear Regression Gradient with Regularisation

Parameters:
	params      = vector containing rolled up X and Θ
		X       = array of features for each movie
		Θ       = array of weights for each user
	gradient    = vector containing rolled up Xgrad and ΘGrad from previous iteration
	Y           = array of ratings for each movie
	R           = array of the existance of ratings for each movie
	numUsers    = number of users
	numMovies   = number of movies
	numFeatures = number of features
	λ           = regularisation constant
	
Modifies:
	gradient = XGrad and ΘGrad are updated based on the given weights and regularisation costant
"""
function grad!(params, gradient, Y, numUsers, numMovies, numFeatures, λ)
	@printf "hi"
	# Unfold params -> X, Θ
	X, Θ = unroll(params, numUsers, numMovies, numFeatures)
	
	XGrad = zeros(size(X))
	ΘGrad = zeros(size(Θ))

	for n = 1:size(Y)[2]
		i = Y[2, n]
		j = Y[1, n]
		@devec x = X[i, :]
		@devec þ = Θ[j, :]
		temp = dot(x[:], þ[:]) - Y[3, n]
		@devec XGrad[i, :] += temp .* þ + λ.*x
		@devec ΘGrad[j, :] += temp .* x + λ.*þ
	end

	gradient[:] = rollup(XGrad, ΘGrad)
end

function costGrad!(params, gradient, Y, numUsers, numMovies, numFeatures, λ)
	# Unfold params -> X, Θ
	X, Θ = unroll(params, numUsers, numMovies, numFeatures)
	
	XGrad = zeros(size(X))
	ΘGrad = zeros(size(Θ))

	# Calculate cost
	J = 0
	start = rand(1:size(Y)[2] - 5_000_000)
	for n = start:start+5_000_000
		i = Y[2, n]
		j = Y[1, n]
		@devec x = X[i, :]
		@devec þ = Θ[j, :]
		temp = dot(x[:], þ[:]) - Y[3, n]
		J += temp^2
		@devec XGrad[i, :] += temp .* þ + λ.*x
		@devec ΘGrad[j, :] += temp .* x + λ.*þ
	end
	J = 1/2 * J
	# add regularisation
	J += λ/2 * (sum(Θ.^2) + sum(X.^2))

	gradient[:] = rollup(XGrad, ΘGrad)

	return J
end

"""
Unroll parameter vector into X and Θ

Parameters:
	params = vector containing rolled up X and Θ
	numUsers    = number of users
	numMovies   = number of movies
	numFeatures = number of features

Returns:
	X = array of features for each movie
	Θ = array of weights for each user
"""
function unroll(params, numUsers, numMovies, numFeatures)
	X = reshape(params[1 : numMovies * numFeatures], numMovies, numFeatures)
	Θ = reshape(params[numMovies * numFeatures + 1 : end], numUsers, numFeatures)
	return X, Θ
end

"""
Roll up X and Θ into single vector

Parameters:
	X = array of features for each movie
	Θ = array of weights for each user

Returns:
	params = vector containing rolled up X and Θ
"""
function rollup(X, Θ)
	return params = vcat(X..., Θ...)
end

"""
Entry point

Params:
	dataPath = path to() CSVs
"""
function main()
	dataPath = ""
	if ARGS[1] == "small"
		dataPath = "../ml-latest-small/fixed/"
	elseif ARGS[1] == "full"
		dataPath = "../ml-latest/fixed/"
	end

	# Load data from CSV - using wc to speed up loading by preallocating
	# N.B. requires cat and wc to be installed - a shell should not be nessecary though
	lines = parse(readall(pipeline(`cat $[dataPath]ratings.csv`, `wc -l`)))
	@time ratings = readdlm(dataPath * "ratings.csv", ',', header=true, dims=(lines, 4))
	ratingsHeader = ratings[2]
	ratings = ratings[1]

	lines = parse(readall(pipeline(`cat $[dataPath]movies.csv`, `wc -l`)))
	@time movies = readdlm(dataPath * "movies.csv", ',', header=true, dims=(lines, 3))
	moviesHeader = movies[2]
	movies = movies[1]

	lines = parse(readall(pipeline(`cat X-results.csv`, `wc -l`)))
	@time X = readdlm("X-results.csv", ',', header=false, dims=(lines, 18))

	lines = parse(readall(pipeline(`cat Theta-results.csv`, `wc -l`)))
	@time Θ = readdlm("Theta-results.csv", ',', header=false, dims=(lines, 18))

	lines = parse(readall(pipeline(`cat $[dataPath]train.csv`, `wc -l`)))
	@time train = readdlm(dataPath * "train.csv", ',', header=true, dims=(lines, 4))[1]

	lines = parse(readall(pipeline(`cat $[dataPath]test.csv`, `wc -l`)))
	@time test = readdlm(dataPath * "test.csv", ',', header=true, dims=(lines, 4))[1]

	@printf "Finished loading, begin processing\n"

	train[:, 3] *= 2
	train = Array{Int}(train[:, 1:3]')
	test[:, 3] *= 2
	test = Array{Int}(test[:, 1:3]')

	# Count users, movies and features
	numUsers = length(union(ratings[:,1]))
	numMovies = size(movies)[1]

	numFeatures = 18 # Leave this for now, 18 initiall chosen b/c 18 listed genres
	λ = 0.05
	
	@printf "Finished processing, begin learning\n"

	function costTemp(x)
		cost(x, train, numUsers, numMovies, numFeatures, λ)	
	end

	function gradTemp!(x, y)
		grad!(x, y, train, numUsers, numMovies, numFeatures, λ)
	end

	i = 0
	function costGradTemp!(x, y)
		i += 1
		if i % 500 == 0
			@printf "%i/%i \t %f \t %s\n" i 20 * 1000 costTemp(x) now()
		end
		costGrad!(x, y, train, numUsers, numMovies, numFeatures, λ)
	end

	@time @show results = optimize(DifferentiableFunction(costTemp, gradTemp!, costGradTemp!), rollup(X,Θ), iterations=1000)

	@printf "Finished learning, begin saving\n"

	writecsv("X-results.csv", X)
	writecsv("Theta-results.csv", Θ)

	@show score = cost(results.minimum, test, numUsers, numMovies, numFeatures, 0) / size(test)[2]
	
	X, Θ = unroll(results.minimum, numUsers, numMovies, numFeatures)

	f = open("score.txt", "w")
	write(f, "$(score)", "\n")
	close(f)

	@printf "Done :D\n"
end

main()
