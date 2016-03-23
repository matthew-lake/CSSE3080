using Images
using Colors
using HDF5
using FixedPointNumbers

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

function process(list, movies, X, name)
	@printf "Start %s\n" name
	numImages = size(list)[1] * 3

	h5 = h5open("$(name).hdf5", "w")
	dset_data = d_create(h5, "data", datatype(Float32), dataspace(90, 120, 3, numImages))
	dset_label = d_create(h5, "label", datatype(Float32), dataspace(size(X)[2], numImages))

	idx = 0
	for i in list
		if idx % 1000 == 0
			@printf "%i/%i\t%s\n" idx numImages now()
		end

		index = findfirst(movies[:, 1] .== i)
		title = movies[index, 2]

		if isfile("movies/$(safe(title))/1.jpg") && 
			isfile("movies/$(safe(title))/2.jpg") && 
			isfile("movies/$(safe(title))/3.jpg")
			for j = 1:3
				img = load("movies/$(safe(title))/$(j).jpg")
				data = convert(Array, separate(img))
				if size(data)[1] < 90 || size(data)[2] < 120 || size(data)[3] < 3
					@show title
					continue
				else
					data = convert(Array, separate(img))[1:90, 1:120, 1:3]
				end
				#data = Array{UInt8}(round(data * 255))
				idx += 1
				h5["label"][1:size(X)[2], idx] = X[index, :]
				h5["data"][map(x -> 1:x, size(data))..., idx] = data
			end
		end
	end

	close(h5)
	@printf "End %s\n" name
end

function main(dataPath)
	lines = parse(readall(pipeline(`cat $[dataPath]movies.csv`, `wc -l`)))
	@time movies = readdlm(dataPath * "movies.csv", ',', header=true, dims=(lines, 3))
	movies = movies[1]
	
	lines = parse(readall(pipeline(`cat X-results.csv`, `wc -l`)))
	@time X = readdlm("X-results.csv", ',', header=false, dims=(lines, 18))

	function exclude(x)
		index = x
		title = movies[index, 2]
		if isfile("movies/$(safe(title))/1.jpg") && 
			isfile("movies/$(safe(title))/2.jpg") && 
			isfile("movies/$(safe(title))/3.jpg")
			return true
		else
			return false
		end
	end

	validMovies = filter(exclude, movies[:, 1])
	order = shuffle(collect(1:size(validMovies)[1]))
	validMovies = validMovies[order, :]

	trainEnd = Int(floor(size(validMovies)[1] * 0.8))
	trainMovies = validMovies[1:trainEnd, :]
	crossEnd = Int(floor(size(validMovies)[1] * 0.9))
	crossMovies = validMovies[trainEnd+1 : crossEnd, :]
	testMovies = validMovies[crossEnd+1 : end, :]

	@printf "Finished loading\n"
	
	process(trainMovies, movies, X, "train")
	process(crossMovies, movies, X, "cross")
	process(testMovies, movies, X, "test")
		
	@printf "Done :)\n"
end

main("../ml-latest/fixed/")
