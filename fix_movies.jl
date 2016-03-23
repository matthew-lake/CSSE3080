using Gadfly

function main(dataPath)
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

	newMovies = deepcopy(movies[:,1])
	newRatings = deepcopy(ratings[:,2])
	newLinks = deepcopy(links[:,1])
	newTags = deepcopy(tags[:,2])

	@printf "Finished loading, going into loop\n"
	@printf "%s/%s" 0 length(movies[:,1])

	for i = 1:length(movies[:, 1])
		if i % 1_000 == 0
			@printf "\n%s/%s" i length(movies[:,1])
		elseif i % 100 == 0
			@printf "."
		end

		if movies[i,1] != i
			newMovies[i] = i
			newRatings[ratings[:,2] .== movies[i,1]] = i
			newLinks[links[:,1] .== movies[i,1]] = i
			newTags[tags[:,2] .== movies[i,1]] = i
		end
	end
	
	@printf "\nFinished loop, moving changes into main array\n"

	movies[:,1] = newMovies
	ratings[:,2] = newRatings
	links[:,1] = newLinks
	tags[:,2] = newTags

	movies = [moviesHeader; movies]
	ratings = [ratingsHeader; ratings]
	links = [linksHeader; links]
	tags = [tagsHeader; tags]

	@printf "Finished moving, begin saving\n"

	if !isdir(dataPath * "fixed")
		mkdir(dataPath * "fixed")
	end
	writecsv(dataPath * "fixed/movies.csv", movies)
	writecsv(dataPath * "fixed/ratings.csv", ratings)
	writecsv(dataPath * "fixed/links.csv", links)
	writecsv(dataPath * "fixed/tags.csv", tags)

	@printf "Done :)\n"
end

main("ml-latest/fixed/")
