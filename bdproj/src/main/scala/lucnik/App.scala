package lucnik

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.log4j.{Level, Logger, Priority}

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

    Logger.getRootLogger.setLevel(Level.WARN)

    Logger.getRootLogger.log(Level.DEBUG, s"Loading data from: $dataLocation")

    //    val bookRDD = sc.textFile(dataLocation)

    val data = sc.textFile(dataLocation)
    val numAs = data.filter(line => line.contains("a")).count()
    val numBs = data.filter(line => line.contains("b")).count()

    println(s"Lines with a: $numAs, Lines with b: $numBs")
  }
}
