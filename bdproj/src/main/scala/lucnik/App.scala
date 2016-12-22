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
            "UniqueCarrier",
            //"FlightNum",
            //"TailNum",
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

        df.printSchema
        df.show

        println("Dropping cancelled flights")
        df = df.filter(df("Cancelled") === 0)
            .drop("Cancelled")

        println("Casting columns to double")
        for (colName <- Array("DepTime", "ArrDelay", "DepDelay", "TaxiOut", "CRSElapsedTime", "CRSArrTime", "Distance")) {
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
            val distinctCount = df.select(countDistinct(colName)).head.getLong(0)
            println(s"\tDistinct values for $colName: $distinctCount")
            if (distinctCount == 1) {
                println(s"\t\t => Dropping $colName")
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


        val bucketSize = 250
        println(s"Bucketizing Distance (bucketSize = $bucketSize")
        val distMax = (df.select(max($"Distance")).head.getDouble(0) + 1).asInstanceOf[Int]
        var buckets = Array[Double](0)
        for (i <- bucketSize to distMax + bucketSize by bucketSize) {
            buckets :+= i.asInstanceOf[Double]
        }

        val bucketizer = new Bucketizer()
            .setInputCol("Distance")
            .setOutputCol("DistanceBucket")
            .setSplits(buckets)

        // Transform original data into its bucket index.
        df = bucketizer.transform(df).drop("Distance").withColumnRenamed("DistanceBucket", "Distance")

        // Actually it is easier to operate directly on the columns instead of using Bucketizer, if there's no need of a pipeline
        println("Converting hhmm times to hour buckets")
        for (colName <- Array("DepTime", "CRSDepTime", "CRSArrTime")) {
            println(s"\tConverting $colName")
            if (df.columns contains colName) {
                df = df.withColumn(colName, expr(s"cast($colName / 100 as int)"))
            }
        }

        println("Using OneHotEncoders for categorical variables")
        for (colName <- Array("Origin", "Dest", "Year", "Month", "DayOfMonth", "DayOfWeek", "DayOfYear", "UniqueCarrier")) {
            if (df.columns contains colName) {
                println(s"\tTransforming $colName")
                val indexer = new StringIndexer()
                    .setInputCol(colName)
                    .setOutputCol(colName + "Index")
                    .fit(df)
                val indexed = indexer.transform(df)

                val encoder = new OneHotEncoder()
                    .setInputCol(colName + "Index")
                    .setOutputCol(colName + "Vec")
                df = encoder.transform(indexed)

                df = df.drop(colName)
                    .withColumnRenamed(colName + "Vec", colName)
                    .drop(colName + "Index")
            }
        }

        println(s"Numble of examples: " + df.count())

        df.printSchema
        df.show

        var assembler = new VectorAssembler()
            .setInputCols(df.drop("ArrDelay").columns)
            .setOutputCol("features")

        var data = assembler.transform(df).select("features", "ArrDelay")
        data = data.withColumnRenamed("ArrDelay", "label")

        println("Feature selection")
        val selector = new ChiSqSelector()
            .setNumTopFeatures(10)
            .setFeaturesCol("features")
            .setOutputCol("selectedFeatures")
            .fit(data)

        println("Index of selected features: " + selector.selectedFeatures.mkString(", "))

        data = selector.transform(data)

        data.printSchema()
        data.show()


        println("Training Classifier")

        // Index labels, adding metadata to the label column.
        // Fit on whole dataset to include all labels in index.
        val labelIndexer = new StringIndexer()
            .setInputCol("label")
            .setOutputCol("indexedLabel")
            .fit(data)

        // Automatically identify categorical features, and index them.
        // Set maxCategories so features with > 4 distinct values are treated as continuous.
        var featureIndexer = new VectorIndexer()
            .setInputCol("selectedFeatures")
            .setOutputCol("indexedFeatures")
            .fit(data)

        // Split the data into training and test sets (30% held out for testing).
        var Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), 42)

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
            .addGrid(rfr.maxDepth, Array(5, 15))
            .build()

        val regressionEvaluator = new RegressionEvaluator()
            .setLabelCol("label")
            .setPredictionCol("prediction")
            .setMetricName("rmse")

        val cv = new CrossValidator()
            .setEstimator(pipeline)
            .setEvaluator(regressionEvaluator)
            .setEstimatorParamMaps(paramGrid)
            .setNumFolds(5)

        // Run cross-validation, and choose the best set of parameters.
        val cvModel = cv.fit(trainingData)

        val bestPipelineModel = cvModel.bestModel.asInstanceOf[PipelineModel]
        val stages = bestPipelineModel.stages
        val rfrStage = stages.last.asInstanceOf[RandomForestRegressionModel]
        println("Best model params:")
        println("\tNumber of trees for the best model: " + rfrStage.getNumTrees)
        println("\tMaxDepth for the best model: " + rfrStage.getMaxDepth)
        println()

        println("Features " + rfrStage.featureImportances)

        // Make predictions.
        var predictions = cvModel.transform(testData)

        // Select example rows to display.
        predictions.select("prediction", "label", "features").show

        // Select (prediction, true label) and compute test error.
        var rmse = regressionEvaluator.evaluate(predictions)
        var r2 = regressionEvaluator
            .setMetricName("r2")
            .evaluate(predictions)

        println(s"Root Mean Squared Error (RMSE) on test data = $rmse")
        println(s"R^2 on test data: $r2")
    }
}
