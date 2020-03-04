import org.apache.spark.SparkContext._

import scala.io._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd._
import org.apache.log4j.Logger
import org.apache.log4j.Level

import scala.util.Random


case class Data(idx: Long, name: String, totalFunding : Double, rounds : Double, seed : Double, venture : Double, roundA: Double, roundB: Double)
case class DataEncode(idx: Long, totalFunding : Int, rounds : Int, seed : Int, venture : Int, roundA: Int, roundB: Int,
                      crowdFunding: Int, angel: Int, privateEquity: Int)
case class Classification(idx: Long, status : String)

class KNN(var neighbors: Int) extends Serializable {
  // Fit a K-Nearest-Neighbors model with Spark RDD's
  def fit(X: RDD[Data], y : RDD[Classification], Xtest: RDD[Data]) : Unit = {
    val y_indexed = y.map(line => (line.idx, line))
    val XY_indexed = X.map(line => (line.idx,line)).join(y_indexed)
    Xtest.cartesian(XY_indexed)
      .map({case (a, b) => (a, (b._2._2.status, distance(a, b._2._1)))}) // Get each row's distance to every other row
      .sortBy({case (_, b) => b._2}) // Sort by lowest distance first
      .groupByKey() // Group by row
      .mapValues(v => v.take(neighbors)) // Get the N nearest neighbor categories
      .map({case (a, b) => (a, get_category(b.toList))}) // Get the most likely category for each row
      .collect()
      .foreach(println)
  }

//  def predict(Xtest: RDD[Data]): Array[Double] = {
//    // Predict X test
//  }

  // Gets the most likely category from a list of category-distance pairs
  def get_category(distances : List[(String, Double)]) : String = {
    distances.map({case (category, _) => category}) // Get the category of each category-distance pair
      .groupBy(identity) // Get all possible categories for each row
      .mapValues(_.size) // Get the frequency of each possible category
      .maxBy(_._2)._1 // Get the category that is most frequent for this row
  }

  // Calculate the euclidean distance between two observations
  def distance(a : Data, b : Data) : Double = {
    var dist = 0.0;
    dist += Math.pow(a.totalFunding - b.totalFunding, 2.0)
    dist += Math.pow(a.rounds - b.rounds, 2.0)
    dist += Math.pow(a.seed - b.seed, 2.0)
    dist += Math.pow(a.venture - b.venture, 2.0)
    dist += Math.pow(a.roundA - b.roundA, 2.0)
    dist += Math.pow(a.roundB - b.roundB, 2.0)
    Math.sqrt(dist)
  }
}

object KNN {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val conf = new SparkConf().setAppName("KNN").setMaster("local[4]")
    val sc = new SparkContext(conf)

    val data = sc.textFile("./data/investments.csv")
      .map(_.split(",(?=([^\\\"]*\\\"[^\\\"]*\\\")*[^\\\"]*$)")).zipWithIndex()
      .map({case(line, idx) => Data(idx, line(1), getTotalFunding(line(5).trim()), line(11).toDouble, line(18).toDouble,
        line(19).toDouble, line(32).toDouble, line(33).toDouble)}).sample(false, 0.01)

    val categories = sc.textFile("./data/investments.csv")
      .map(_.split(",(?=([^\\\"]*\\\"[^\\\"]*\\\")*[^\\\"]*$)")).zipWithIndex()
      .map({case(line,idx) => Classification(idx, line(6))}).sample(false ,0.01)

    val normalized_data = normalize(data)

    val model = new KNN(9)

    val split_data = train_test_split(normalized_data, categories, 0.5)

    val Xtrain = split_data._1
    val ytrain = split_data._2
    val Xtest = split_data._3
    val ytest = split_data._4

    model.fit(Xtrain, ytrain, Xtest)
  }

  // Split data and classifications into training and testing data
  def train_test_split(X: RDD[Data], y: RDD[Classification], frac: Double):
  (RDD[Data], RDD[Classification], RDD[Data], RDD[Classification]) = {
    val Xtrain = X.sample(withReplacement = false, frac) // Get a sample of data to train on
    val Xtrain_idx = Xtrain.map(line => (line.idx, line)) // Get the indexes of the training set
    val ytrain = y.map(line => (line.idx, line))
      .join(Xtrain_idx).map({case(_, pair) => pair._1}) // Get a sample of classifications to train on
    val Xtest = X.subtract(Xtrain) // All non-training data is test data
    val ytest = y.subtract(ytrain) // All non-training classifications are test classifications

    (Xtrain, ytrain, Xtest, ytest)
  }

  // Get the total funding from a weirdly formatted string
  def getTotalFunding(funding: String) : Double = {
    if (funding == "-") {
      return 0
    }
    funding.replaceAll(",", "")
      .replaceAll("^\"+|\"+$", "")
      .trim().toDouble
  }

  // Normalize data with minmax technique
  def normalize(data: RDD[Data]) : RDD[Data] = {
    val maxFunding = data.takeOrdered(1)(Ordering[Double].reverse.on(_.totalFunding))(0).totalFunding
    val minFunding = data.takeOrdered(1)(Ordering[Double].on(_.totalFunding))(0).totalFunding
    val maxRounds = data.takeOrdered(1)(Ordering[Double].reverse.on(_.rounds))(0).rounds
    val minRounds = data.takeOrdered(1)(Ordering[Double].on(_.rounds))(0).rounds
    val maxSeed = data.takeOrdered(1)(Ordering[Double].reverse.on(_.seed))(0).seed
    val minSeed = data.takeOrdered(1)(Ordering[Double].on(_.seed))(0).seed
    val maxVenture = data.takeOrdered(1)(Ordering[Double].reverse.on(_.venture))(0).venture
    val minVenture = data.takeOrdered(1)(Ordering[Double].on(_.venture))(0).venture
    val maxA = data.takeOrdered(1)(Ordering[Double].reverse.on(_.roundA))(0).roundA
    val minA = data.takeOrdered(1)(Ordering[Double].on(_.roundA))(0).roundA
    val maxB = data.takeOrdered(1)(Ordering[Double].reverse.on(_.roundB))(0).roundB
    val minB = data.takeOrdered(1)(Ordering[Double].on(_.roundB))(0).roundB

    data.map(rec => Data(rec.idx, rec.name,
      (rec.totalFunding - minFunding) / (maxFunding - minFunding),
      (rec.rounds - minRounds) / (maxRounds - minRounds),
      (rec.seed - minSeed) / (maxSeed - minSeed),
      (rec.venture - minVenture) / (maxVenture - minVenture),
      (rec.roundA - minA) / (maxA - minA),
      (rec.roundB - minB) / (maxB - minB)))
  }
}