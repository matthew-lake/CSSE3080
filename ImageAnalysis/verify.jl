using Requests
import Requests: get, post, put, delete, options

function safe(name)
	result = replace(name, ":", " ")
	result = replace(result, "\\", " ")
	result = replace(result, "/", " ")
	result = replace(result, "*", " ")
	result = replace(result, "?", " ")
	result = replace(result, "|", " ")
	result = replace(result, "<", " ")
	result = replace(result, ">", " ")
	result = replace(result, "\"", " ")
	result = replace(result, r"[^\x20-\x7e]", "")
	result = strip(result)
	return result
end

function urlsafe(title)
	result = replace(title, r"[^\x20-\x7e]", "")
	result = strip(result)
	result = replace(result, " ", "+")
	result = replace(result, ",", "")
	result = replace(result, "!", "")
	result = replace(result, "'", "")
	result = replace(result, "-", "")
	return result
end

function main(dataPath)
	lines = parse(readall(pipeline(`cat $[dataPath]movies.csv`, `wc -l`)))
	@time movies = readdlm(dataPath * "movies.csv", ',', header=true, dims=(lines, 3))
	movies = movies[1]

	@printf "Finished loading, going into loop\n"

	count = 0
	for i = 1:length(movies[:, 1])
		title = movies[i, 2]

		if !isfile("movies/$(safe(title))/1.jpg") || 
			!isfile("movies/$(safe(title))/2.jpg")	|| 
			!isfile("movies/$(safe(title))/3.jpg")
			count += 1
		end
	end

	@show count
	
	@printf "Done :)\n"
end

main("../ml-latest/fixed/")
