package lucnik

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, IntegerType}


object App {
    def main(args: Array[String]) {
        var dataLocation = ""
        dataLocation = args(0)

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

        var df = spark.read
            .format("com.databricks.spark.csv")
            .option("header", "true")
            .option("inferSchema", "true") // only infer int if there's no NA
            .csv(dataLocation)

        val columnNames = Seq(
            "Year",
            "Month",
            "DayofMonth",
            "DayOfWeek",
            "DepTime",
            "CRSDepTime",
            //"ArrTime", // forbidden
            "CRSArrTime",
            //"UniqueCarrier",  // mmm
            //"FlightNum", // mmm
            //"TailNum",  // mmm
            //"ActualElapsedTime", // forbidden
            "CRSElapsedTime",
            //"AirTime", // forbidden
            "ArrDelay", // target
            "DepDelay", // wat?
            "Origin",
            "Dest",
            "Distance",
            //"TaxiIn", // forbidden
            "TaxiOut",
            "Cancelled" // what with this?
            //"CancellationCode", // what with this?
            //"Diverted", // forbidden
            //"CarrierDelay", // forbidden
            //"WeatherDelay", // forbidden
            //"NASDelay", // forbidden
            //"SecurityDelay", // forbidden
            //"LateAircraftDelay" // forbidden
        )

        df = df.select(columnNames.head, columnNames.tail: _*)

        // could not be needed the print
        df.printSchema
        df.show(10)

        df.select(df("DayOfWeek")).distinct.show()
        df.select(df("TaxiOut")).distinct.show()

        // TODO what to do with cancelled flights?
        df.filter(df("Cancelled") === 1)
            .select("Cancelled", "TaxiOut", "DepTime")
            .distinct()
            .show()

        df = df.filter(df("Cancelled") === 0)
        df = df.drop("Cancelled")

        for (colName <- Array("DepTime", "ArrDelay", "DepDelay", "TaxiOut")) {
            df = df.withColumn(colName, df.col(colName).cast(DoubleType))
        }

        //for (colName <- df.columns) {
        //    df.select(df(colName)).distinct.show()
        //}

        for (colName <- df.columns) {
            df.select(count(df.col(colName).isNull)).show
        }

        df.select(min("ArrDelay"), max("ArrDelay")).show

        df.printSchema

        print("Converting hhmm times to hour buckets")
        for (colName <- Array("DepTime", "CRSDepTime", "CRSArrTime")) {
            df = df.withColumn(colName, expr(s"cast($colName / 100 as int)"))
        }

        print("One Hot Encoding")
        for (colName <- Array("Origin", "Dest")) {
            val indexer = new StringIndexer()
                .setInputCol(colName)
                .setOutputCol(colName + "Index")
                .fit(df)
            val indexed = indexer.transform(df)

            val encoder = new OneHotEncoder()
                .setInputCol(colName + "Index")
                .setOutputCol(colName + "Vec")
            df = encoder.transform(indexed)

            df = df.withColumn(colName, df.col(colName + "Vec"))
                .drop(colName + "Index", colName + "Vec")

            // TODO remove this line
            df = df.drop(colName)
        }

        df.printSchema()
        df.show()

        val assembler = new VectorAssembler()
            //.setInputCols(df.columns.drop(df.columns.indexOf("ArrDelay")))
            .setInputCols(df.columns)
            .setOutputCol("features")
        val training = assembler.transform(df)

        training.printSchema()
        training.show()

        //val model = new LogisticRegression()
        //    .setLabelCol("ArrDelay")
        //    .setMaxIter(10)

        val model = new LinearRegression()
            .setFeaturesCol("features")
            .setLabelCol("ArrDelay")
            .setMaxIter(10)
            .setRegParam(0.3)
            .setElasticNetParam(0.8)

        val trainedModel = model.fit(training)

        println(s"Coefficients: ${trainedModel.coefficients} Intercept: ${trainedModel.intercept}")

        val trainingSummary = trainedModel.summary
        println(s"numIterations: ${trainingSummary.totalIterations}")
        println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
        trainingSummary.residuals.show()
        println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
        println(s"r2: ${trainingSummary.r2}")
    }
}


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
