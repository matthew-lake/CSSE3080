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
	count = Int(floor(size(Y)[2] * 0.1))
	start = rand(1:size(Y)[2] - count)
	for n = start:start+count
	#for n = 1:size(Y)[2]
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
	@time ratings = readdlm(dataPath * "ratings.csv", ',', header=true, dims=(lines, 4))[1]

	lines = parse(readall(pipeline(`cat $[dataPath]train.csv`, `wc -l`)))
	@time train = readdlm(dataPath * "train.csv", ',', header=true, dims=(lines, 4))[1]

	lines = parse(readall(pipeline(`cat $[dataPath]cross.csv`, `wc -l`)))
	@time cross = readdlm(dataPath * "cross.csv", ',', header=true, dims=(lines, 4))[1]

	lines = parse(readall(pipeline(`cat $[dataPath]movies.csv`, `wc -l`)))
	@time movies = readdlm(dataPath * "movies.csv", ',', header=true, dims=(lines, 3))
	moviesHeader = movies[2]
	movies = movies[1]

	@printf "Finished loading, begin processing\n"

	train[:, 3] *= 2
	train = Array{Int}(train[:, 1:3]')
	cross[:, 3] *= 2
	cross = Array{Int}(cross[:, 1:3]')

	# Count users, movies and features
	numUsers = length(union(ratings[:,1]))
	numMovies = size(movies)[1]

	numFeatures = 18
	λ = 0.05
	X = rand(numMovies, 18)
	Θ = rand(numUsers, 18)

	if numFeatures == 18
		for i = 1:numMovies
			X[i, 1] = contains(movies[i, 3], "Action") ? 2 : 0
			X[i, 2] = contains(movies[i, 3], "Adventure") ? 2 : 0
			X[i, 3] = contains(movies[i, 3], "Animation") ? 2 : 0
			X[i, 4] = contains(movies[i, 3], "Children") ? 2 : 0
			X[i, 5] = contains(movies[i, 3], "Comedy") ? 2 : 0
			X[i, 6] = contains(movies[i, 3], "Crime") ? 2 : 0
			X[i, 7] = contains(movies[i, 3], "Documentary") ? 2 : 0
			X[i, 8] = contains(movies[i, 3], "Drama") ? 2 : 0
			X[i, 9] = contains(movies[i, 3], "Fantasy") ? 2 : 0
			X[i, 10] = contains(movies[i, 3], "Film-Noir") ? 2 : 0
			X[i, 11] = contains(movies[i, 3], "Horror") ? 2 : 0
			X[i, 12] = contains(movies[i, 3], "Musical") ? 2 : 0
			X[i, 13] = contains(movies[i, 3], "Mystery") ? 2 : 0
			X[i, 14] = contains(movies[i, 3], "Romance") ? 2 : 0
			X[i, 15] = contains(movies[i, 3], "Sci-Fi") ? 2 : 0
			X[i, 16] = contains(movies[i, 3], "Thriller") ? 2 : 0
			X[i, 17] = contains(movies[i, 3], "War") ? 2 : 0
			X[i, 18] = contains(movies[i, 3], "Western") ? 2 : 0
		end
	end
	scoreboard = Array{Any}(0,3)
	for i = 0.1:0.10:1.00
		tempTrain = train[:, 1:Int(floor(i*size(train)[2]))]

		function costTemp(x)
			cost(x, train, numUsers, numMovies, numFeatures, λ)	
		end

		function gradTemp!(x, y)
			grad!(x, y, train, numUsers, numMovies, numFeatures, λ)
		end

		function costGradTemp!(x, y)
			costGrad!(x, y, tempTrain, numUsers, numMovies, numFeatures, λ)
		end

		tic()
		@time @show results = optimize(DifferentiableFunction(costTemp, gradTemp!, costGradTemp!), rollup(X,Θ), iterations=10)
		time = toc()

		@show score = cost(results.minimum, tempTrain, numUsers, numMovies, numFeatures, 0) / size(tempTrain)[2]
		@show score = cost(results.minimum, cross, numUsers, numMovies, numFeatures, 0) / size(cross)[2]
		scoreboard = [scoreboard; λ score time]
	end

	#writecsv("lambba-scoreboard.csv", scoreboard)

	@printf "Done :D\n"
end

main()
