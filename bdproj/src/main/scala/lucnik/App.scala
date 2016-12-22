package lucnik

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression, MultilayerPerceptronClassifier, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression.{LinearRegression, RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.functions.{countDistinct, _}
import org.apache.spark.sql.types.{DoubleType, IntegerType}


object App {
    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("Big Data Project")
        val sc = new SparkContext(conf)

        // TODO do we need to enable Hive support?
        val spark = SparkSession
            .builder()
            .appName("")
            .enableHiveSupport()
            .getOrCreate()

        import spark.implicits._

        val dataLocation = args(0)
        // TODO debug
        //println("DATA LOCATION IS FIXED")
        //val dataLocation = "./data/2008_1000.csv"

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
            "DayOfMonth",
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
            "DepDelay",
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

        // could not be needed the print
        df.printSchema
        df.show

        println("Dropping cancelled flights")
        df = df.filter(df("Cancelled") === 0)
            .drop("Cancelled")

        println("Casting columns to double")
        // TODO shall we cast every column?
        for (colName <- Array("DepTime", "ArrDelay", "DepDelay", "TaxiOut", "CRSElapsedTime", "Distance")) {
            df = df.withColumn(colName, df.col(colName).cast(DoubleType))
        }

        println("Dropping flights with null values for ArrDelay")
        df = df.filter(df("ArrDelay").isNotNull)

        println("Creating column DayOfYear")
        df = df.withColumn("DayOfYear", dayofyear(concat_ws("-", $"Year", $"Month", $"DayOfMonth")))

        println("Dropping columns which have only null values")
        var onlyNullValues = Array[String]()
        for (colName <- df.columns) {
            if (df.select(df.col(colName)).count == df.filter(df.col(colName).isNull).count) {
                println(s"\t Dropping $colName")
                onlyNullValues +:= colName
            }
        }
        df = df.drop(onlyNullValues: _*)

        println("Describing dataset")
        df.describe().show

        println("Dropping columns with only one categorical value")
        var singleCatValue = Array[String]()
        for (colName <- df.columns) {
            if (df.select(countDistinct(colName)).head.getLong(0) == 1) {
                println(s"\t Dropping $colName")
                singleCatValue +:= colName
            }
        }
        df = df.drop(singleCatValue: _*)

        df.printSchema

        println("Computing correlation coefficients")
        for (colName <- Array("DayOfWeek", "DepTime", "CRSDepTime", "CRSArrTime", "CRSElapsedTime", "DepDelay", "Distance", "TaxiOut")) {
            if (df.columns contains colName) {
                val corr = df.stat.corr(colName, "ArrDelay")
                println(s"\t$corr => $colName")
            }
        }

        df.createOrReplaceTempView("df")

        println("Showing values for each column")
        for (colName <- df.columns) {
            spark.sql(s"SELECT $colName, COUNT($colName) AS cnt " +
                s"FROM df " +
                s"GROUP BY $colName").sort($"$colName".desc)
        }

        println("Checking for null values")
        for (colName <- df.columns) {
            df.select(sum(df.col(colName).isNull.cast(IntegerType))).show
        }

        println("Drop all NA!!!")
        df = df.na.drop()

        println("Converting hhmm times to hour buckets")
        for (colName <- Array("DepTime", "CRSDepTime", "CRSArrTime")) {
            println(s"\tConverting $colName")
            if (df.columns contains colName) {
                df = df.withColumn(colName, expr(s"cast($colName / 100 as int)"))
            }
        }

        println("Using OneHotEncoders for categorical variables")
        for (colName <- Array("Origin", "Dest", "Year", "Month", "DayOfMonth", "DayOfWeek", "DayOfYear")) {
            if (df.columns contains colName) {
                println(s"\tTransforming $colName")
                val indexer = new StringIndexer()
                    .setInputCol(colName)
                    .setOutputCol(colName + "Index")
                    .fit(df)
                val indexed = indexer.transform(df)
                //df = indexer.transform(df)

                val encoder = new OneHotEncoder()
                    .setInputCol(colName + "Index")
                    .setOutputCol(colName + "Vec")
                df = encoder.transform(indexed)

                df = df.drop(colName)
                    .withColumnRenamed(colName + "Vec", colName)
                    .drop(colName + "Index")

                //df = df.drop(colName)
                //    .withColumnRenamed(colName + "Index", colName)
            }
        }

        println(s"Numble of examples: " + df.count())

        df.printSchema
        df.show

        var assembler = new VectorAssembler()
            //.setInputCols(df.columns.drop(df.columns.indexOf("ArrDelay")))
            .setInputCols(df.drop("ArrDelay").columns)
            .setOutputCol("features")

        var data = assembler.transform(df).select("features", "ArrDelay")
        data = data.withColumnRenamed("ArrDelay", "label")

        data.printSchema()
        data.show()

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


        println("Trainin Classifier")
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
            .fit(data)

        // Split the data into training and test sets (30% held out for testing).
        var Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), 42)
        //
        ////// Train a RandomForest model.
        //val trainer = new RandomForestClassifier()
        //    .setLabelCol("indexedLabel")
        //    .setFeaturesCol("indexedFeatures")
        //    .setMaxBins(200)
        //    .setNumTrees(10)

        // specify layers for the neural network:
        // input layer of size 4 (features), two intermediate of size 5 and 4
        // and output of size 3 (classes)
        //val layers = Array[Int](df.columns.length - 1, 75, 100, data.select(countDistinct("label")).head.getLong(0).asInstanceOf[Int])
        // create the trainer and set its parameters
        //val trainer = new MultilayerPerceptronClassifier()
        //    .setLabelCol("indexedLabel")
        //    .setFeaturesCol("indexedFeatures")
        //    //.setFeaturesCol("features")
        //    .setLayers(layers)
        //    .setBlockSize(128)
        //    .setSeed(1234L)
        //    .setMaxIter(1000)
        //
        //// Convert indexed labels back to original labels.
        //val labelConverter = new IndexToString()
        //    .setInputCol("prediction")
        //    .setOutputCol("predictedLabel")
        //    .setLabels(labelIndexer.labels)
        //
        //// Chain indexers and forest in a Pipeline.
        //var pipeline = new Pipeline()
        //    //.setStages(Array(labelIndexer, featureIndexer, trainer, labelConverter))
        //    .setStages(Array(featureIndexer, labelIndexer, trainer, labelConverter))

        //var td = labelIndexer.transform(data)
        //td = featureIndexer.transform(td)
        //val lm = trainer.fit(td)
        //
        //var ted = labelIndexer.transform(testData)
        //ted = featureIndexer.transform(ted)
        //
        //val result = lm.transform(ted)
        //val predictionAndLabels = result.select("prediction", "label")
        //val evaluator = new MulticlassClassificationEvaluator()
        //    .setMetricName("accuracy")
        //println("Accuracy: " + evaluator.evaluate(predictionAndLabels))
        //var rmse = regressionEvaluator.evaluate(predictionAndLabels)
        //println("Root Mean Squared Error (RMSE) on test data = " + rmse)


        //// Train model. This also runs the indexers.
        //var model = pipeline.fit(trainingData)
        //
        //// Make predictions.
        //var predictions = model.transform(testData)
        //
        //// Select example rows to display.
        //predictions.select("predictedLabel", "label", "features").show
        //
        //val multiClassEvaluator = new MulticlassClassificationEvaluator()
        //    .setLabelCol("indexedLabel")
        //    .setPredictionCol("prediction")
        //    .setMetricName("accuracy")
        //val accuracy = multiClassEvaluator.evaluate(predictions)
        //println("Accuracy = " + accuracy)
        ////println("Test Error = " + (1.0 - accuracy))
        //
        //val regressionEvaluator = new RegressionEvaluator()
        //    .setLabelCol("label")
        //    .setPredictionCol("prediction")
        //    .setMetricName("rmse")
        //
        //var rmse = regressionEvaluator.evaluate(predictions)
        //println("Root Mean Squared Error (RMSE) on test data = " + rmse)

        //val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
        //println("Learned classification forest model:\n" + rfModel.toDebugString)

        // Train a RandomForestRegressor model.
        println(s"Random forest regressor")
        val rfr = new RandomForestRegressor()
            .setLabelCol("label")
            .setFeaturesCol("indexedFeatures")

        // Chain indexer and forest in a Pipeline.
        var pipeline = new Pipeline()
            .setStages(Array(featureIndexer, rfr))

        // We use a ParamGridBuilder to construct a grid of parameters to search over.
        // With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
        // this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
        val paramGrid = new ParamGridBuilder()
            .addGrid(rfr.numTrees, Array(10, 50, 100))
            .build()

        val regressionEvaluator = new RegressionEvaluator()
            .setLabelCol("label")
            .setPredictionCol("prediction")
            .setMetricName("rmse")

        val cv = new CrossValidator()
            .setEstimator(pipeline)
            .setEvaluator(regressionEvaluator)
            .setEstimatorParamMaps(paramGrid)
            .setNumFolds(5) // Use 3+ in practice

        // Run cross-validation, and choose the best set of parameters.
        val cvModel = cv.fit(trainingData)

        println("Best models params :" + cvModel.bestModel.explainParams()))

        // Make predictions.
        var predictions = cvModel.transform(testData)

        // Select example rows to display.
        predictions.select("prediction", "label", "features").show


        // Select (prediction, true label) and compute test error.
        var rmse = regressionEvaluator.evaluate(predictions)
        var r2 = regressionEvaluator
            .setMetricName("r2")
            .evaluate(predictions)

        //var regressor = model.stages.last.asInstanceOf[RandomForestRegressionModel]

        println("Root Mean Squared Error (RMSE) on test data = " + rmse)
        println(s"R^2 on test data: $r2")
        //println("Learned regression forest model:\n" + regressor.toDebugString)


        //val rfModel = model.stages(1).asInstanceOf[RandomForestRegressionModel]
        //println("Learned regression forest model:\n" + rfModel.toDebugString)

        //
        //df.printSchema
        //
        //assembler = new VectorAssembler()
        //    //.setInputCols(df.columns.drop(df.columns.indexOf("ArrDelay")))
        //    .setInputCols(df.drop("TaxiOut").columns)
        //    .setOutputCol("features")
        //
        //data = assembler.transform(df)
        //data = data.withColumn("label", data.col("ArrDelay"))
        //
        //featureIndexer = new VectorIndexer()
        //    .setInputCol("features")
        //    .setOutputCol("indexedFeatures")
        //    .setMaxCategories(4)
        //    .fit(data)
        //
        //
        //val Array(trainingData2, testData2) = data.randomSplit(Array(0.7, 0.3), 42)
        //
        //pipeline = new Pipeline()
        //    .setStages(Array(featureIndexer, rfr))
        //
        //model = pipeline.fit(trainingData2)
        //
        //// Make predictions.
        //predictions = model.transform(testData2)
        //
        //// Select example rows to display.
        //predictions.select("prediction", "label", "features").show(5)
        //
        //rmse = regressionEvaluator.evaluate(predictions)
        //
        //r2 = regressionEvaluator
        //    .setMetricName("r2")
        //    .evaluate(predictions)
        //
        //println("Root Mean Squared Error (RMSE) on test data = " + rmse)
        //println(s"R^2 on test data: $r2")
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
