package lucnik

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.functions._


object App {
  def main(args: Array[String]) {

    var dataLocation = ""
    if (args.length > 0) {
      dataLocation = args(0)
    }
    else {
      dataLocation = "./dataset.csv"
    }

    val conf = new SparkConf().setAppName("Big Data Project")
    val sc = new SparkContext(conf)

    // TODO do we need to enable Hive support?
    val spark = SparkSession
      .builder()
      .appName("")
      .enableHiveSupport()
      .getOrCreate()

    import spark.implicits._

    Logger.getRootLogger.setLevel(Level.WARN)
    Logger.getRootLogger.log(Level.DEBUG, s"Loading data from: $dataLocation")

    var data: DataFrame = spark.read
      .option("header", "true").
	  .option("inferSchema", "true").  // does this line infer types automatically?
      .csv(dataLocation)

    val columnNames = Seq(
      "Year",
      "Month",
      "DayofMonth",
      "DayOfWeek",
      "DepTime",
      "CRSDepTime",
      //      "ArrTime", // forbidden
      "CRSArrTime",
      "UniqueCarrier",
      "FlightNum",
      "TailNum",
      //      "ActualElapsedTime", // forbidden
      "CRSElapsedTime",
      //      "AirTime", // forbidden
      "ArrDelay", // target
      "DepDelay",
      "Origin",
      "Dest",
      "Distance",
      //      "TaxiIn", // forbidden
      "TaxiOut",
      "Cancelled",
      "CancellationCode",
      "Diverted"
      //      "CarrierDelay", // forbidden
      //      "WeatherDelay", // forbidden
      //      "NASDelay", // forbidden
      //      "SecurityDelay", // forbidden
      //      "LateAircraftDelay" // forbidden
    )

    data = data.select(columnNames.head, columnNames.tail: _*)

    // could not be needed the print
    print(data.printSchema)
    print(data.show(10))
  }
}

//
// // Exercise 3 by Jesús
//  val spark = SparkSession
//     .builder()
//     .appName("")
//     .getOrCreate()
//
// var df = spark.read
//     .format("com.databricks.spark.csv")
//     .option("sep", " ")
//     .option("header", "false")
//     .load("file:///filepath.csv")
//     .withColumnRenamed("_c0", "project_name")
//     .withColumnRenamed("_c1", "page_title")
//     .withColumnRenamed("_c2", "num_requests")
//     .withColumnRenamed("_c3", "content_size")
//     .select(
//         col("_c0").as("project_name"),
//         col("_c1").as("page_title"),
//         col("_c2").as("long").as("num_requests"),
//         col("_c3").as("long").as("content_size"),
//     )
//
// val project_summary = df
//     .groupBy("project_name")
//     .agg(
//         count("page_title").as("num_pages"),
//         sum("content_size").as("content_size"),
//         avg("num_requests").as("mean_requests")
//     )
//
// project_summary.show
//
// val most_visited = df.join(
//     project_summary.select("project_name", "num_requests"),
//     "project_name"
// ).filter(col("num_requests") > col(mean_requests))
//
// most_visited.show
