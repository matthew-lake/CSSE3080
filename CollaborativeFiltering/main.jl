using Optim

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
function cost(params, Y, R, numUsers, numMovies, numFeatures, λ)
	# Unfold params -> X, Θ
	X, Θ = unroll(params, numUsers, numMovies, numFeatures)

	# Calculate cost
	J = 1/2 * sum(R .* (X * Θ' - Y).^2) 
	# add regularisation
	J += λ/2 * (sum(Θ.^2) + sum(X.^2))

	if rand() > 0.95
		@show J
	end

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
function grad!(params, gradient, Y, R, numUsers, numMovies, numFeatures, λ)
	# Unfold params -> X, Θ
	X, Θ = unroll(params, numUsers, numMovies, numFeatures)
	
	XGrad = zeros(size(X))
	ΘGrad = zeros(size(Θ))

	#batchSize = min(1000, numMovies - 1)
	#start = batchSize >= numMovies ? 1 : rand(1:numMovies - batchSize)
	# Calculate XGrad
	#for i = start:start + batchSize
	for i = 1:numMovies
		idx = find(R[i, :])
		ΘTemp = Θ[idx, :]
		YTemp = Y[i, idx];
		XGrad[i, :] = (X[i, :] * ΘTemp' - YTemp) * ΘTemp + λ * X[i,:];
	end

	#batchSize = min(200, numUsers - 1)
	#start = batchSize >= numUsers ? 1 : rand(1:numUsers - batchSize)
	# Calculate ΘGrad
	#for i = start:start + batchSize
	for i = 1:numUsers
		idx = find(R[:, i])
		XTemp = X[idx, :];
		YTemp = Y[idx, i];
		ΘGrad[i, :] = XTemp' * (XTemp * Θ[i, :]' - YTemp) + λ * Θ[i, :]';
	end

	gradient[:] = rollup(XGrad, ΘGrad)
end

function costGrad!(params, gradient, Y, R, numUsers, numMovies, numFeatures, λ)
	# Unfold params -> X, Θ
	X, Θ = unroll(params, numUsers, numMovies, numFeatures)
	
	XGrad = zeros(size(X))
	ΘGrad = zeros(size(Θ))
	
	start = rand(1:size(X)[1]-200)

	#tic()
	# Calculate XGrad
	for i = start:start+200
		idx = find(R[i, :])
		ΘTemp = Θ[idx, :]
		YTemp = Y[i, idx];
		XGrad[i, :] = (X[i, :] * ΘTemp' - YTemp) * ΘTemp + λ * X[i,:];
	end

	#toc()
	start = rand(1:size(Θ)[1]-200)
	#tic()
	# Calculate ΘGrad
	for i = start:start+200
		idx = find(R[:, i])
		XTemp = X[idx, :];
		YTemp = Y[idx, i];
		ΘGrad[i, :] = XTemp' * (XTemp * Θ[i, :]' - YTemp) + λ * Θ[i, :]';
	end

	#toc()
	#tic()
	gradient[:] = rollup(XGrad, ΘGrad)

	# Calculate cost
	J = 1/2 * sum(R .* (X * Θ' - Y).^2) 
	# add regularisation
	J += λ/2 * (sum(Θ.^2) + sum(X.^2))

	if rand() > 0.99
		@show J
	end

	#toc()
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
function main(dataPath)
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

	lines = parse(readall(pipeline(`cat $[dataPath]links.csv`, `wc -l`)))
	@time links = readdlm(dataPath * "links.csv", ',', header=true, dims=(lines, 3))
	linksHeader = links[2]
	links = links[1]

	lines = parse(readall(pipeline(`cat $[dataPath]tags.csv`, `wc -l`)))
	@time tags = readdlm(dataPath * "tags.csv", ',', header=true, dims=(lines, 4))
	tagsHeader = tags[2]
	tags = tags[1]

	# Count users, movies and features
	numUsers = length(union(ratings[:,1]))
	numMovies = size(movies)[1]
	numFeatures = 18 # Leave this for now, 18 initiall chosen b/c 18 listed genres

	# Random initialisation
	X = rand(numMovies, numFeatures)
	Θ = rand(numUsers, numFeatures)
	
	Y = zeros(Int8, numMovies, numUsers)
	for i = 1:size(ratings)[1]
		Y[Int(ratings[i,2]), Int(ratings[i,1])] = Int8(ratings[i,3] * 2)
	end

	R = Y .!= 0

	@time @show cost(rollup(X, Θ), Y, R, numUsers, numMovies, numFeatures, 0.1)
	@time grad!(rollup(X, Θ), rollup(zeros(X), zeros(Θ)),Y, R, numUsers, numMovies, numFeatures, 0.1)

	@time @show results = optimize(x -> cost(x, Y, R, numUsers, numMovies, numFeatures, 0.1), (x,y) -> grad!(x,y, Y, R, numUsers, numMovies, numFeatures, 0.1), rollup(X,Θ), iterations=10_000)
	#@time @show optimize(DifferentiableFunction(x -> cost(x, Y, R, numUsers, numMovies, numFeatures, 0.1), (x,y) -> grad!(x,y, Y, R, numUsers, numMovies, numFeatures, 0.1), (x,y) -> costGrad!(x,y, Y, R, numUsers, numMovies, numFeatures, 0.1)), rollup(X,Θ), iterations=100)
#=
	gradient = rollup(ones(size(X)), ones(size(Θ)))
	params = rollup(X, Θ)
	α = 0.0001
	tic()
	for i = 1:10_000
		grad!(params, gradient, Y, R, numUsers, numMovies, numFeatures, 0.1)
		params -= α * gradient
		params[1:10]
		if i % 100 == 0
			toc()
			gradient[1:10]
			@show cost(params, Y, R, numUsers, numMovies, numFeatures, 0.1)
			tic()
		end
	end=#

	X, Θ = unroll(results.minimum, numUsers, numMovies, numFeatures)
	@show X[1,:] * Θ[1,:]'
	@show Y[1,1]
end

main("../ml-latest-small/fixed/")
