import org.apache.spark.SparkContext._
import scala.io._
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.rdd._
import org.apache.log4j.Logger
import org.apache.log4j.Level
import scala.util.Random


case class Data(idx: Long, sepLen : Double, sepWid : Double, petLen : Double, petWid : Double)
case class Category(idx: Long, category : String) {
  override def toString : String = category
}

class KNN(var neighbors: Int) extends Serializable {
  // Fit a K-Nearest-Neighbors model with Spark RDD's
  def fit(X: RDD[Data], y : RDD[Category], Xtest: RDD[Data]) : Array[(Data, String)] = {
    val y_indexed = y.map(line => (line.idx, line))
    val XY_indexed = X.map(line => (line.idx,line)).join(y_indexed)
    Xtest.cartesian(XY_indexed)
      .map({case (a, b) => (a, (b._2._2.category, distance(a, b._2._1)))}) // Get each row's distance to every other row
      .sortBy({case (_, b) => b._2}) // Sort by lowest distance first
      .groupByKey() // Group by row
      .mapValues(v => v.take(neighbors)) // Get the N nearest neighbor categories
      .map({case (a, b) => (a, get_category(b.toList))}) // Get the most likely category for each row
      .collect()
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

    val conf = new SparkConf().setAppName("KNN").setMaster("local[4]")
    val sc = new SparkContext(conf)

    val data = sc.textFile("./data/iris.csv")
      .map(_.split(",")).zipWithIndex()
      .map({case(line, idx) => Data(idx, line(0).toDouble, line(1).toDouble, line(2).toDouble, line(3).toDouble)})

    val categories = sc.textFile("./data/iris.csv")
      .map(_.split(",")).zipWithIndex()
      .map({case(line,idx) => Category(idx, line(4))})

    val model = new KNN(9)

    val split_data = train_test_split(data, categories, 0.5)

    val Xtrain = split_data._1
    val ytrain = split_data._2
    val Xtest = split_data._3
    val ytest = split_data._4

    val result = model.fit(Xtrain, ytrain, Xtest)
    result.foreach(println)

    // accuracy
    println(accuracy(result, ytest.collect()))

    // precision
    println(precision(result, ytest.collect()))

    // recall
    println(recall(result, ytest.collect()))

    //f1
    println(f1_score(result, ytest.collect()))
  }

  def train_test_split(X: RDD[Data], y: RDD[Category], frac: Double): (RDD[Data], RDD[Category], RDD[Data], RDD[Category]) = {
    val Xtrain = X.sample(false, frac);
    val Xtrain_idx = Xtrain.map(line => (line.idx, line))
    val ytrain = y.map(line => (line.idx, line)).join(Xtrain_idx).map({ case (idx, pair) => pair._1 })
    val Xtest = X.subtract(Xtrain)
    val ytest = y.subtract(ytrain)

    return (Xtrain, ytrain, Xtest, ytest)
  }

  def accuracy(result : Array[(Data, String)], ytest : Array[Category]) : Double = {
    val ypred_tuple = result.map({case( data, pred) => (data.idx, pred)})
    val ytest_tuple = ytest.map(line => (line.idx, line.category))
    val ypred_ytest = ( ypred_tuple ++ ytest_tuple ).
      groupBy( _._1 ).
      map(list_tuple_pair => list_tuple_pair._2).
      map(tuple_pair => (tuple_pair(0), tuple_pair(1)))


    val correct_classifications = ypred_ytest.count({
      case(pred, actual) => pred._2 == actual._2
    })

    val total_classifications = result.length

    val accuracy = correct_classifications * 1.0 / total_classifications


    return accuracy
  }

  def precision(result : Array[(Data, String)], ytest : Array[Category]) : scala.collection.mutable.Map[String, Double] = {
    val categories = result.map(pred => pred._2).distinct
    val precision_map = scala.collection.mutable.Map[String, Double]()
    val category_strings = categories.map(_.toString).distinct

    val ypred_tuple = result.map({case( data, pred) => (data.idx, pred)})
    val ytest_tuple = ytest.map(line => (line.idx, line.category))
    val ypred_ytest = ( ypred_tuple ++ ytest_tuple ).
      groupBy( _._1 ).
      map(list_tuple_pair => list_tuple_pair._2).
      map(tuple_pair => (tuple_pair(0), tuple_pair(1)))

    for (cat <- category_strings) {
      // filter results for those PREDICTED to be in this category
      val filtered = ypred_ytest.filter({
        case (pred, actual) => pred._2 == cat
      })

      // filter results where prediction for this category was correct (True Positive)
      val true_positives = filtered.filter({
        case (pred, actual) => pred._2 == actual._2
      })


      val precision = true_positives.size * 1.0 / filtered.size

      precision_map += (cat -> precision)
    }
    return precision_map
  }

  def recall(result : Array[(Data, String)], ytest : Array[Category]) : scala.collection.mutable.Map[String, Double] = {
    val categories = result.map(pred => pred._2).distinct
    val recall_map = scala.collection.mutable.Map[String, Double]()
    val category_strings = categories.map(_.toString).distinct

    val ypred_tuple = result.map({case( data, pred) => (data.idx, pred)})
    val ytest_tuple = ytest.map(line => (line.idx, line.category))
    val ypred_ytest = ( ypred_tuple ++ ytest_tuple ).
      groupBy( _._1 ).
      map(list_tuple_pair => list_tuple_pair._2).
      map(tuple_pair => (tuple_pair(0), tuple_pair(1)))

    for (cat <- category_strings) {
      // filter results for those PREDICTED to be in this category
      val filtered = ypred_ytest.filter({
        case (pred, actual) => actual._2 == cat
      })

      // filter results where prediction for this category was correct (True Positive)
      val true_positives = filtered.filter({
        case (pred, actual) => pred._2 == actual._2
      })


      val precision = true_positives.size * 1.0 / filtered.size

      recall_map += (cat -> precision)
    }
    return recall_map
  }

  def f1_score(result : Array[(Data, String)], ytest : Array[Category]) : scala.collection.mutable.Map[String, Double] = {
    val precisionScore = precision(result, ytest)
    val recallScore = recall(result, ytest)
    val categories = result.map(pred => pred._2).distinct
    val category_strings = categories.map(_.toString).distinct
    val f1_map = scala.collection.mutable.Map[String, Double]()
    for (cat <- category_strings) {
      f1_map += (cat -> (2*precisionScore(cat)*recallScore(cat))/(precisionScore(cat)+recallScore(cat)))
    }

    return f1_map
  }
}