using Optim
using Devectorize

function cost(Y::Array{Int32}, users, start, finish)
	count = finish - start
	temp = @parallel (+) for i = start:finish
		user = Y[1, i]
		movie = Y[2, i]
		rating = Y[3, i]
		
		others = find(Y[2, :] .== movie)
		others = Y[1, others]
		@show result = (predict(movie, user, others, Y, users) - rating)^2
		f = open("pearson_results/$(i).txt", "w")
		write(f, "$(result)", "\n")
		close(f)
		result
	end
	return temp / count
end

@everywhere function predict(movie::Int32, me::Int32, others::Array{Int32}, Y::Array{Int32}, users)
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
function main(dataPath)
	# Load data from CSV - using wc to speed up loading by preallocating
	# N.B. requires cat and wc to be installed - a shell should not be nessecary though
	lines = parse(readall(pipeline(`cat $[dataPath]ratings.csv`, `wc -l`)))
	@time ratings = readdlm(dataPath * "ratings.csv", ',', header=true, dims=(lines, 4))
	ratingsHeader = ratings[2]
	ratings = ratings[1]

	ratings[:, 3] *= 2
	Y = Array{Int32}(ratings[:, 1:3]')

	#numUsers = length(union(Y[1, :]))
	#@time users = pmap(x -> find(Y[1,:] .== x), collect(1:numUsers))
	#writecsv("users.csv", users)
	
	users = Vector{Vector{Int}}()
	f = open("users.csv")
	i = 0
	@time while (l = readline(f)) != nothing && strip(l) != ""
		i += 1
		if i % 10_000 == 0
			@printf "%i/%i\t%s\n" i 300_000 now()
		end
		@time l = replace(l, "[", "")
		@time l = replace(l, "]", "")
		@time temp = map(parse, split(l, ","))
		@time push!(users, temp)
	end

	@printf "Read csv\n"

	count = parse(ARGS[1])
	start = rand(1:size(Y)[2] - count)
	@time @show cost(Y, users, start, start+count)
end

main("../ml-latest/fixed/")
