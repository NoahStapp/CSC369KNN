import org.apache.spark.SparkContext._
import scala.io._
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.rdd._
import org.apache.log4j.Logger
import org.apache.log4j.Level

case class Data(sepLen : Double, sepWid : Double, petLen : Double, petWid : Double, category : String)
case class Category(category : String)

class KNN(var neighbors: Int) extends Serializable {
  def fit(X: RDD[Data], y : RDD[Category], Xtest: RDD[Data]) : Unit = {
    Xtest.cartesian(X)
      .map({case (a, b) => (a, (b.category, distance(a, b)))})
      .sortBy(_._2._2)
      .groupByKey()
      .mapValues(v => v.take(neighbors))
      .collect()
      .map({case (x,y) => (x, y.map(row => row._1).groupBy(identity).mapValues(_.size).maxBy(_._2)._1)})
      .foreach(println)
  }
//  def predict(Xtest: RDD[Data]): Array[Double] = {
//    // Predict X test
//  }
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
      .map(_.split(","))
      .map(line => Data(line(0).toDouble, line(1).toDouble, line(2).toDouble, line(3).toDouble, line(4)))

    val categories = sc.textFile("./data/iris.csv")
      .map(_.split(","))
      .map(line => Category(line(4)))

    val model = new KNN(9)

    model.fit(data, categories, data)
  }
}