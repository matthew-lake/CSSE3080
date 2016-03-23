using Devectorize
using Gadfly

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

function predict(movie, me, others, Y, users)
	meIndices = find(Y[1, :] .== me)
	meMovies = Y[2, meIndices]
	weightSum = 0.0
	summation = 0.0
	otherIndices = Int[]
	tempMeIndices = Int[]
	for other in others
		otherIndices = users[other]
		otherMovies = Y[2, otherIndices]
		jointMovies = intersect(meMovies, otherMovies)
		if length(jointMovies) == 0
			continue
		end
		tempMeIndices = filter(x -> in(Y[2, x], jointMovies), meIndices)
		tempOtherIndices = filter(x -> in(Y[2, x], jointMovies), otherIndices)

		meRatings = Y[3, tempMeIndices]
		meAverage = mean(meRatings)
		otherRating = Y[3, find(Y[2, otherIndices] .== movie)][1]
		otherRatings = Y[3, tempOtherIndices]
		otherAverage = mean(otherRatings)

		temp = sqrt(sum((meRatings - meAverage).^2)) * sqrt(sum((otherRatings - otherAverage).^2))
		if temp == 0
			continue
		end

		@devec weight = sum((meRatings - meAverage).*(otherRatings - otherAverage)) ./ temp

		summation += (otherRating - otherAverage) * weight
		weightSum += weight
	end

	meRatings = Y[3, meIndices]
	meAverage = mean(meRatings)

	return meAverage + summation / weightSum
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
	#lines = parse(readall(pipeline(`cat $[dataPath]ratings.csv`, `wc -l`)))
	#@time ratings = readdlm(dataPath * "ratings.csv", ',', header=true, dims=(lines, 4))[1]

	#lines = parse(readall(pipeline(`cat $[dataPath]train.csv`, `wc -l`)))
	#@time train = readdlm(dataPath * "train.csv", ',', header=true, dims=(lines, 4))[1]

	lines = parse(readall(pipeline(`cat $[dataPath]cross.csv`, `wc -l`)))
	@time cross = readdlm(dataPath * "cross.csv", ',', header=true, dims=(lines, 4))[1]

	lines = parse(readall(pipeline(`cat $[dataPath]test.csv`, `wc -l`)))
	@time test = readdlm(dataPath * "test.csv", ',', header=true, dims=(lines, 4))[1]

	lines = parse(readall(pipeline(`cat $[dataPath]movies.csv`, `wc -l`)))
	@time movies = readdlm(dataPath * "movies.csv", ',', header=true, dims=(lines, 3))
	moviesHeader = movies[2]
	movies = movies[1]
	
	lines = parse(readall(pipeline(`cat X-results.csv`, `wc -l`)))
	@time X = readdlm("X-results.csv", ',', header=false, dims=(lines, 18))

	lines = parse(readall(pipeline(`cat Theta-results.csv`, `wc -l`)))
	@time Θ = readdlm("Theta-results.csv", ',', header=false, dims=(lines, 18))

	testUsersData = readcsv("test-users-data.csv")
	users = testUsersData[:, 2]
	usersNumMovies = testUsersData[:, 3]
	usersAvgRatings = testUsersData[:, 4]
	usersRatingsDifference = testUsersData[:, 5]

	#=@printf "users\n"
	usersIndices = Vector{Vector{Int}}()
	f = open("usersIndices.csv")
	i = 0
	@time while (l = readline(f)) != nothing && strip(l) != ""
		i += 1
		if i % 10_000 == 0
			@printf "%i/%i\t%s\n" i 240_000 now()
		end
		l = replace(l, "[", "")
		l = replace(l, "]", "")
		temp = map(parse, split(l, ","))
		if typeof(temp) != Array{Int, 1}
			temp = []
		end
		push!(usersIndices, temp)
	end
	close(f)=#

	#data = [train; cross]
	#data[:, 3] *= 2
	#data = Array{Int}(data[:, 1:3]')
	test[:, 3] *= 2
	test = Array{Int}(test[:, 1:3]')

	#nonEmptyUserIndices = filter(x -> length(usersIndices[x]) > 0, collect(1:size(usersIndices)[1]))
	#realUsers = map(x -> data[1, nonEmptyUserIndices[x][1]], collect(1:size(nonEmptyUserIndices)[1]))
	#reallyNonEmptyUserIndices = usersIndices[nonEmptyUserIndices]
	
	function userCostModel(user, indices)
		cost = 0
		for i in indices
			cost += (dot(X[test[2, i], :][:], Θ[test[1, i], :][:]) - test[3, i])^2
		end
		return cost / length(indices)
	end

	#=function userCostMemory(user, indices)
		cost = 0
		for i in indices
			movie = test[2, i]
			rating = test[3, i]
		
			others = find(data[2, :] .== movie)
			others = data[1, others]
			others = intersect(others, realUsers)
			cost += (predict(movie, user, others, data, reallyNonEmptyUserIndices) - rating)^2
		end
		return cost / length(indices)
	end=#

	#memoryCost = zeros(length(realUsers))
	modelCost = zeros(length(users))
	idx = 0
	for user in users
		if idx % 1000 == 0
			@show idx
		end
		idx += 1
		indices = find(test[1, :] .== user)
		try
			modelCost[idx] = userCostModel(user, indices)
			#memoryCost[idx] = userCostMemory(user, indices)
		catch e
			@show e
			continue
		end
	end

	#writecsv("memoryCost.csv", memoryCost)
	writecsv("modelCost.csv", modelCost)
	#writecsv("final-out.csv", hcat(users, usersNumMovies, usersAvgRatings, usersRatingsDifference, memoryCost, modelCost))

	#=@printf "movies\n"
	moviesIndices = Vector{Vector{Int}}()
	f = open("moviesIndices.csv")
	i = 0
	@time while (l = readline(f)) != nothing && strip(l) != ""
		i += 1
		if i % 1_000 == 0
			@printf "%i/%i\t%s\n" i 33_000 now()
		end
		l = replace(l, "[", "")
		l = replace(l, "]", "")
		temp = map(parse, split(l, ","))
		if typeof(temp) != Array{Int, 1}
			temp = []
		end
		push!(moviesIndices, temp)
	end
	close(f)

	@printf "Finished loading, begin processing\n"

	data = [train; cross]
	data[:, 3] *= 2
	data = Array{Int}(data[:, 1:3]')
	test[:, 3] *= 2
	test = Array{Int}(test[:, 1:3]')

	users = union(test[1, :])
	@time usersAvgRatings = pmap(x -> mean(data[2, usersIndices[x]]), users)
	@show size(usersAvgRatings)
	@show size(usersIndices)
	@show size(usersIndices[1])
	@show usersIndices[1]
	@show data[2, usersIndices[1]]
	@show mean(data[2, usersIndices[1]])
	@show map(x -> mean(data[2, usersIndices[x]]), collect(1:5))
	@show size(data[2, usersIndices[1]])
	# Count users, movies and features
	numUsers = length(union(ratings[:,1]))
	numMovies = size(movies)[1]

	numFeatures = 18
	λ = 0.05

	function costTemp(x)
		cost(x, train, numUsers, numMovies, numFeatures, λ)	
	end

	#@time moviesIndices = pmap(x -> find(data[2, :] .== x), movies[:, 1])
	@time moviesRatingsCount = pmap(x -> length(moviesIndices[x]), movies[:, 1])
	@time moviesAvgRating = pmap(x -> mean(data[3, moviesIndices[x]]), movies[:, 1])

	users = union(test[1, :])
	@time usersNumMovies = pmap(x -> length(usersIndices[x]), users)
	@time usersAvgRatings = pmap(x -> mean(moviesRatingsCount[data[2, usersIndices[x]]]), users)
	@time usersRatingsDifference = pmap(x -> mean(map(y -> moviesAvgRating[data[2, y]] - data[3, y], usersIndices[x])), users)

	@printf "Done calculating, saving\n"

	writecsv("test-users-data.csv", hcat(users, usersNumMovies, usersAvgRatings, usersRatingsDifference))

	@printf "Done :D\n"=#	
end

main()
