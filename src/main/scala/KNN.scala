import org.apache.spark.SparkContext._
import scala.io._
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.rdd._
import org.apache.log4j.Logger
import org.apache.log4j.Level

import scala.util.Random


case class Data(idx: Long, sepLen : Double, sepWid : Double, petLen : Double, petWid : Double, category : String)
case class Category(idx: Long, category : String)

class KNN(var neighbors: Int) extends Serializable {
  // Fit a K-Nearest-Neighbors model with Spark RDD's
  def fit(X: RDD[Data], y : RDD[Category], Xtest: RDD[Data]) : Unit = {
    Xtest.cartesian(X)
      .map({case (a, b) => (a, (b.category, distance(a, b)))}) // Get each row's distance to every other row
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
    dist += Math.pow(a.sepLen - b.sepLen, 2.0)
    dist += Math.pow(a.sepWid - b.sepWid, 2.0)
    dist += Math.pow(a.petLen - b.petLen, 2.0)
    dist += Math.pow(a.petWid - b.petWid, 2.0)
    dist
  }
}

object KNN {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val conf = new SparkConf().setAppName("Asgn5").setMaster("local[4]")
    val sc = new SparkContext(conf)

    val data = sc.textFile("./data/iris.csv")
      .map(_.split(",")).zipWithIndex()
      .map({case(line, idx) => Data(idx, line(0).toDouble, line(1).toDouble, line(2).toDouble, line(3).toDouble, line(4))})

    val categories = sc.textFile("./data/iris.csv")
      .map(_.split(",")).zipWithIndex()
      .map({case(line,idx) => Category(idx,line(4))})

    val model = new KNN(9)

    val split_data = train_test_split(data, categories, 0.5)

    val Xtrain = split_data._1
    val ytrain = split_data._2
    val Xtest = split_data._3
    val ytest = split_data._4

    model.fit(Xtrain, ytrain, Xtest)
  }

  def train_test_split(X: RDD[Data], y: RDD[Category], frac: Double): (RDD[Data],RDD[Category], RDD[Data],RDD[Category]) = {
    val Xtrain = X.sample(false, frac);
    val Xtrain_idx = Xtrain.map(line => (line.idx, line))
    val ytrain = y.map(line => (line.idx, line)).join(Xtrain_idx).map({case(idx, pair) => pair._1})
    val Xtest = X.subtract(Xtrain)
    val ytest = y.subtract(ytrain)

    return (Xtrain, ytrain, Xtest, ytest);
  }



}