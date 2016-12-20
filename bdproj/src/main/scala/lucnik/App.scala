package lucnik

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression.{LinearRegression, RandomForestRegressor}
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, IntegerType}


object App {
    def main(args: Array[String]) {
        val dataLocation = args(0)

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
            //"UniqueCarrier",  // TODO use it
            //"FlightNum", // TODO think on it
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
            "Cancelled"
            //"CancellationCode",
            //"Diverted", // forbidden
            //"CarrierDelay", // forbidden
            //"WeatherDelay", // forbidden
            //"NASDelay", // forbidden
            //"SecurityDelay", // forbidden
            //"LateAircraftDelay" // forbidden
        )

        df = df.select(columnNames.head, columnNames.tail: _*)
        df.createOrReplaceTempView("df")

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
            .drop("Cancelled")

        for (colName <- Array("DepTime", "ArrDelay", "DepDelay", "TaxiOut", "CRSElapsedTime")) {
            df = df.withColumn(colName, df.col(colName).cast(DoubleType))
        }

        df = df.filter(df("ArrDelay").isNotNull)

        for (colName <- df.columns) {
            //df.select(df(colName)).distinct.show()

            spark.sql(s"SELECT $colName, COUNT($colName) AS cnt " +
                s"FROM df " +
                s"GROUP BY $colName").sort($"$colName".desc).show
        }

        //println("Checking for null values")
        //for (colName <- df.columns) {
        //    val c: DataFrame = df.select(sum(df.col(colName).isNull.cast(IntegerType)))
        //    c.show
        //    //if (c.rdd.map(_ (0).asInstanceOf[Long]).reduce(_ + _) > 0) {
        //    //    c.show
        //    //}
        //}

        println("Drop all NA!!!")
        df = df.na.drop()

        df.select(min("ArrDelay"), max("ArrDelay")).show

        df.printSchema

        println("Converting hhmm times to hour buckets")
        for (colName <- Array("DepTime", "CRSDepTime", "CRSArrTime")) {
            df = df.withColumn(colName, expr(s"cast($colName / 100 as int)"))
        }

        println("One Hot Encoding")
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

        var assembler = new VectorAssembler()
            //.setInputCols(df.columns.drop(df.columns.indexOf("ArrDelay")))
            .setInputCols(df.columns)
            .setOutputCol("features")
        var data = assembler.transform(df)

        data = data.withColumn("label", data.col("ArrDelay"))

        data.printSchema()
        data.show()
        //
        ////val model = new LogisticRegression()
        ////    .setLabelCol("ArrDelay")
        ////    .setMaxIter(10)
        //
        //val model = new LinearRegression()
        //    .setFeaturesCol("features")
        ////    .setLabelCol("ArrDelay")
        //    .setMaxIter(10)
        //    .setRegParam(0.3)
        //    .setElasticNetParam(0.8)
        //
        //val trainedModel = model.fit(data)
        //
        //println(s"Coefficients: ${trainedModel.coefficients} Intercept: ${trainedModel.intercept}")
        //
        //val trainingSummary = trainedModel.summary
        //println(s"numIterations: ${trainingSummary.totalIterations}")
        //println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
        //trainingSummary.residuals.show()
        //println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
        //println(s"r2: ${trainingSummary.r2}")


        println("Random forest classifier")
        // Index labels, adding metadata to the label column.
        // Fit on whole dataset to include all labels in index.
        val labelIndexer = new StringIndexer()
            .setInputCol("label")
            .setOutputCol("indexedLabel")
            .fit(data)

        // Automatically identify categorical features, and index them.
        // Set maxCategories so features with > 4 distinct values are treated as continuous.
        var featureIndexer = new VectorIndexer()
            .setInputCol("features")
            .setOutputCol("indexedFeatures")
            .setMaxCategories(4)
            .fit(data)

        // Split the data into training and test sets (30% held out for testing).
        var Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), 42)

        // Train a RandomForest model.
        val rfc = new RandomForestClassifier()
            .setLabelCol("indexedLabel")
            .setFeaturesCol("indexedFeatures")
            .setNumTrees(10)

        // Convert indexed labels back to original labels.
        val labelConverter = new IndexToString()
            .setInputCol("prediction")
            .setOutputCol("predictedLabel")
            .setLabels(labelIndexer.labels)

        // Chain indexers and forest in a Pipeline.
        var pipeline = new Pipeline()
            .setStages(Array(labelIndexer, featureIndexer, rfc, labelConverter))

        // Train model. This also runs the indexers.
        //var model = pipeline.fit(trainingData)

        // Make predictions.
        //var predictions = model.transform(testData)

        // Select example rows to display.
        //predictions.select("predictedLabel", "label", "features").show(5)

        //val multiClassEvaluator = new MulticlassClassificationEvaluator()
        //    .setLabelCol("indexedLabel")
        //    .setPredictionCol("prediction")
        //    .setMetricName("accuracy")
        //val accuracy = multiClassEvaluator.evaluate(predictions)
        //println("Accuracy = " + accuracy)
        //println("Test Error = " + (1.0 - accuracy))

        //val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
        //println("Learned classification forest model:\n" + rfModel.toDebugString)

        // Train a RandomForest model.
        println("Random forest regressor")
        val rfr = new RandomForestRegressor()
            .setLabelCol("label")
            .setFeaturesCol("indexedFeatures")

        // Chain indexer and forest in a Pipeline.
        pipeline = new Pipeline()
            .setStages(Array(featureIndexer, rfr))

        // Train model. This also runs the indexer.
        var model = pipeline.fit(trainingData)

        // Make predictions.
        var predictions = model.transform(testData)

        // Select example rows to display.
        predictions.select("prediction", "label", "features").show(5)

        // Select (prediction, true label) and compute test error.
        val regressionEvaluator = new RegressionEvaluator()
            .setLabelCol("label")
            .setPredictionCol("prediction")
            .setMetricName("rmse")

        var rmse = regressionEvaluator.evaluate(predictions)

        println("Root Mean Squared Error (RMSE) on test data = " + rmse)

        //val rfModel = model.stages(1).asInstanceOf[RandomForestRegressionModel]
        //println("Learned regression forest model:\n" + rfModel.toDebugString)


        df.printSchema

        assembler = new VectorAssembler()
            //.setInputCols(df.columns.drop(df.columns.indexOf("ArrDelay")))
            .setInputCols(df.drop("TaxiOut").columns)
            .setOutputCol("features")

        data = assembler.transform(df)
        data = data.withColumn("label", data.col("ArrDelay"))

        featureIndexer = new VectorIndexer()
            .setInputCol("features")
            .setOutputCol("indexedFeatures")
            .setMaxCategories(4)
            .fit(data)


        val Array(trainingData2, testData2) = data.randomSplit(Array(0.7, 0.3), 42)

        pipeline = new Pipeline()
            .setStages(Array(featureIndexer, rfr))

        model = pipeline.fit(trainingData2)

        // Make predictions.
        predictions = model.transform(testData2)

        // Select example rows to display.
        predictions.select("prediction", "label", "features").show(5)

        rmse = regressionEvaluator.evaluate(predictions)
        println("Root Mean Squared Error (RMSE) on test data = " + rmse)
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
