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

    // TODO rly???
    import spark.implicits._

    Logger.getRootLogger.setLevel(Level.WARN)
    Logger.getRootLogger.log(Level.DEBUG, s"Loading data from: $dataLocation")

    var data: DataFrame = spark.read
      .option("header", "true")
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
      "ActualElapsedTime",
      "CRSElapsedTime",
      "AirTime",
      "ArrDelay", // target
      "DepDelay",
      "Origin",
      "Dest",
      "Distance",
      "TaxiIn",
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

    print(data.printSchema)
    print(data.show(10))
  }
}
