function main(dataPath)
	lines = parse(readall(pipeline(`cat $[dataPath]ratings.csv`, `wc -l`)))
	@time ratings = readdlm(dataPath * "ratings.csv", ',', header=true, dims=(lines, 4))
	ratingsHeader = ratings[2]
	ratings = ratings[1]

	order = shuffle(collect(1:size(ratings)[1]))
	ratings = ratings[order, :]

	trainEnd = Int(floor(size(ratings)[1] * 0.8))
	train = ratings[1 : trainEnd, :]
	crossEnd = Int(floor(size(ratings)[1] * 0.9))
	cross = ratings[trainEnd+1 : crossEnd, :]
	test = ratings[crossEnd+1 : end, :]

	train = [ratingsHeader; train]
	cross = [ratingsHeader; cross]
	test = [ratingsHeader; test]

	@printf "Finished moving, begin saving\n"

	writecsv(dataPath * "train.csv", train)
	writecsv(dataPath * "cross.csv", cross)
	writecsv(dataPath * "test.csv", test)

	@printf "Done :)\n"
end

main("ml-latest-small/fixed/")
