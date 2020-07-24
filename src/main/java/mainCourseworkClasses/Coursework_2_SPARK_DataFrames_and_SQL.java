package mainCourseworkClasses;

import static org.apache.spark.sql.functions.avg;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.variance;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;

//import static spark.Spark.*;

import org.apache.spark.sql.SparkSession;

import scala.Tuple2;

public class Coursework_2_SPARK_DataFrames_and_SQL {

	// Path strings to load the .csv files
	private static String PATH = "src/main/resources/";
	private static String movies = "movies.csv";
	private static String ratings = "ratings.csv";
	private static String movieGenres = "movieGenres.csv";
	static File f = new File(PATH + movieGenres);

	// Load up the Spark context
	static SparkConf conf = new SparkConf().setMaster("local").setAppName("My App");
	static JavaSparkContext sc = new JavaSparkContext(conf);
	static SparkSession spark = SparkSession.builder().appName("Java Spark SQL MovieLens Queries")
			.config("spark.some.config.option", "some-value").getOrCreate();

	public static void main(String[] args) throws IOException {

		LogManager.getLogger("org").setLevel(Level.ERROR);

		// STEP 1. Prints out the movies schema
		System.out.println("--- STEP 1. Movies schema ---");
		Dataset<Row> moviesDF = loadMoviesMetadata();
		moviesDF.printSchema();

		// STEP 1. Print out the ratings schema
		System.out.println("--- STEP 1. Ratings schema ---");
		Dataset<Row> ratingsDF = loadRatingsMetadata();
		ratingsDF.printSchema();

		// STEP 2. Separate each movie genre and pair it with its movieId
		List<String> ids = moviesDF.select(col("movieId")).as(Encoders.STRING()).collectAsList();
		@SuppressWarnings("serial")
		JavaPairRDD<String, String> pairs = moviesDF.select(col("genres")).as(Encoders.STRING()).toJavaRDD()
				.flatMapToPair(new PairFlatMapFunction<String, String, String>() {
					int i = 0;

					@Override
					public Iterator<Tuple2<String, String>> call(String s) throws Exception {
						List<Tuple2<String, String>> list = new ArrayList<>();
						for (String token : s.split("\\|")) {
							list.add(new Tuple2<String, String>(ids.get(i).toString(), token.toString()));
						}
						i++;
						return list.iterator();
					}
				});

		// STEP 2. Write the pair into a file: movieGenres.csv
        writePairToFile(pairs);

		// STEP 3. Prints out the ratings schema
		System.out.println("--- STEP 3. Movie Genres schema: ---");
		Dataset<Row> movieGenresDF = loadMovieGenresMetadata();
		movieGenresDF.printSchema();
		
		// STEP 3. Print out 50 frames of movieGenresDF transformation: order by movieId (descending)
		System.out.println("--- STEP 3. Print out 50 frames of movieGenresDF transformation: order by movieId (descending) ---");
		movieGenresDF.select(col("movieId"), col("genres")).orderBy(col("movieId").desc()).show(50);
		
		// STEP 4. Reports top 10 most popular genres
		System.out.println("--- STEP 4. Print out 10 most popular genres ---");
		Dataset<Row> genrePopularityRenamed = groupOrderByCountAndRename(movieGenresDF, "genres", "moviesCount");
		genrePopularityRenamed.show(10);
		
		// STEP 5. Highest user ratings for 10 most popular genres
		System.out.println("--- STEP 5. Highest user ratings for 10 most popular genres ---");
		Dataset<Row>  joinedDF =  ratingsDF.join(movieGenresDF, movieGenresDF.col("movieId")
				.equalTo(ratingsDF.col("movieId")))
				.groupBy(col("userId"),col("genres"))
				.count().orderBy(col("count").desc());
		Dataset<Row>  joinedDFDropDuplicates = joinedDF.dropDuplicates("genres");				
		genrePopularityRenamed.limit(10).join(joinedDFDropDuplicates, joinedDFDropDuplicates.col("genres")
				.equalTo(genrePopularityRenamed.limit(10).col("genres"))).select(joinedDFDropDuplicates.col("genres").as("genre"), col("userId")).show();
		
		// STEP 6. Most common genres rated by the highest raters
		System.out.println("--- STEP 6. Most common genres rated by the highest raters ---");
		Dataset<Row> ratingsCountRenamed = groupOrderByCountAndRename(ratingsDF, "userId", "ratingsCount");
		ratingsCountRenamed.limit(10).join(joinedDF, ratingsCountRenamed.col("userId").equalTo(joinedDF.col("userId"))).orderBy(col("count").desc()).dropDuplicates("userId").select(joinedDF.col("userId"), col("ratingsCount"), col("genres").as("mostCommonGenre")).orderBy(col("ratingsCount").desc()).show();

		// STEP 7. Variance and avg rating of highest rated movies
		System.out.println("--- STEP 7. Variance and avg rating of highest rated movies ---");
		Dataset<Row> avgRatings = ratingsDF.groupBy(col("movieId")).agg(avg("rating")).orderBy(col("avg(rating)").desc()).limit(10);
		Dataset<Row> varianceRatings = ratingsDF.groupBy(col("movieId")).agg(variance("rating"));
		Dataset<Row> topRatingsVariance = avgRatings
		.join(varianceRatings, avgRatings.col("movieId").equalTo(varianceRatings.col("movieId"))).select(avgRatings.col("movieId"), col("avg(rating)"), col("var_samp(rating)"));
		topRatingsVariance.show();
				
		spark.stop();
	}

	/***
	 * Creating DataFrames
	 */
	
	//Loads movies database
	private static Dataset<Row> loadMoviesMetadata() {
		return spark.read().option("inferSchema", true).option("header", true).option("multLine", true)
				.option("mode", "DROPMALFORMED").csv(PATH + movies);
	}

	//Loads ratings database
	private static Dataset<Row> loadRatingsMetadata() {
		return spark.read().option("inferSchema", true).option("header", true).option("multLine", true)
				.option("mode", "DROPMALFORMED").csv(PATH + ratings);
	}

	//Loads movie genres database
	private static Dataset<Row> loadMovieGenresMetadata() {
		return spark.read().option("inferSchema", true).option("header", true).option("multLine", true)
				.option("mode", "DROPMALFORMED").csv(PATH + movieGenres);
	}
	
	//Simplified function to take a dataset, group by specific column and rename a column
	private static Dataset<Row> groupOrderByCountAndRename (Dataset<Row> dataset, String col, String newColName) {
		Dataset<Row> tempDataset = dataset.groupBy(col(col)).count().orderBy(col("count").desc());
		return tempDataset.withColumnRenamed("count", newColName);
	}
	
	//Write to file a pair
	private static void writePairToFile (JavaPairRDD<String, String> pairs) throws IOException {
		if (!f.exists() && !f.isDirectory()) {
			FileWriter fw;
			fw = new FileWriter(PATH + movieGenres, true);
			fw.write("movieId,genre\n");
			pairs.collect().forEach(pair -> {
				try {
					fw.write(pair._1().toString() + "," + pair._2().toString() + "\n");
				} catch (IOException e) {
					e.printStackTrace();
				}
			});
			fw.close();
		}
	}
}
