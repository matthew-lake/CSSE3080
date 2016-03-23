using Gadfly

function main()
	csv = readdlm("ml-latest/fixed/ratings.csv", ',', header=true)
	users = csv[1][:,1]
	movies = csv[1][:,2]
	colours = csv[1][:,3]
	csv = 0

	width = length(union(users))
	height = length(union(movies))

	graph = plot(x=users, y=movies, color=colours, Geom.point, 
		Theme(default_point_size=0.1mm, highlight_width=0mm, grid_line_width=0mm),
		Guide.colorkey("Rating"), Guide.xlabel("User"), Guide.ylabel("Movie"), 
		Guide.xticks(ticks=collect(1:width), label=false), Guide.yticks(label=false))

	draw(PNG("movie_user_graph.png", width * 0.01mm, height * 0.01mm), graph)
end

main()
