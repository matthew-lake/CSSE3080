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
	@printf "%s/%s" 0 length(movies[:,1])

	for i = 1:length(movies[:, 1])
		if i % 1_000 == 0
			@printf "\n%s/%s" i length(movies[:,1])
		elseif i % 100 == 0
			@printf "."
		end

		title = movies[i, 2]

		if !isfile("movies/$(safe(title))/3.jpg")
			data = []
			num = 1
			try
				data = Requests.json(get("https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=50&q=$(urlsafe(title))+trailer&key=AIzaSyDRmOXB5UYs51X_1qNO_5iXn_gsc8OBINw"))

				while data["items"][num]["id"]["kind"] != "youtube#video"
					num += 1
				end

				if !isdir("movies/$(safe(title))")
					mkdir("movies/$(safe(title))")
				end

				id = data["items"][num]["id"]["videoId"]
				for j = 1:3
					image = get("https://i.ytimg.com/vi/$(id)/$(j).jpg")
					save(image, "movies/$(safe(title))/$(j).jpg")
				end
			catch ex
				if data != [] && haskey(data, "items") && length(data["items"]) < num
					if isdir("movies/$(safe(title))")
						@show safe(title)
					end
					continue
				end

				@show safe(title)
				#throw(ex)
				i -= 1
				continue
			end
		end
	end
	
	@printf "Done :)\n"
end

main("../ml-latest/fixed/")
