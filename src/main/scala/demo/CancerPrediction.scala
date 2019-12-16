package demo

import org.apache.log4j
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.{MulticlassMetrics, RegressionMetrics}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, when}
import org.apache.spark.sql.types.IntegerType

/**
  * @author ducnguyen
  * @since 12/5/19
  */
object CancerPrediction {

  def main(args: Array[String]): Unit = {
    log4j.Logger.getLogger("org").setLevel(log4j.Level.ERROR)

    val spark = SparkSession.builder().master("local[1]").getOrCreate()

    val data = spark.read.option("header", "true")
      .option("inferSchema", "true")
      .option("delimiter", ",").csv("src/main/scala/dataset/breast_cancer.csv")
    data.printSchema()

    val df = data.withColumn("Bare Nuclei", when(col("Bare Nuclei").equalTo("?"), 5)
      .otherwise(col("Bare Nuclei")))
      .withColumn("Bare Nuclei", col("Bare Nuclei").cast(IntegerType))

    data.printSchema()

    // Set the input columns as the features we want to use
    val assembler = new VectorAssembler().setInputCols(Array(
      "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape",
      "Marginal Adhesion", "Single Epithelial Cell Size", "Bland Chromatin",
      "Normal Nucleoli", "Mitoses")).
      setOutputCol("features")

    // Transform the DataFrame
    val output = assembler.transform(df).selectExpr("Class as label", "features")

    // Split into train and test set
    val Array(training, test) = output.selectExpr("label", "features").
      randomSplit(Array(0.7, 0.3), seed = 12345)

    // Train model
    val rf = new DecisionTreeClassifier().setMaxBins(2).setMaxDepth(0)

    val model = rf.fit(training)
    val results = model.transform(test).select("features", "label", "prediction")

    import spark.implicits._
    val predictionAndLabels = results.
      select("prediction", "label").
      as[(Double, Double)].
      rdd

    // Evaluation
    val regressionMetrics = new RegressionMetrics(predictionAndLabels)
    println(s"RMSE with RegressionMetrics, = sqrt(MSE) = ${regressionMetrics.rootMeanSquaredError}")

    val mMetrics = new MulticlassMetrics(predictionAndLabels)
    val labels = mMetrics.labels

    println("Accuracy: " + mMetrics.accuracy)
    println("Confusion matrix Actual\\Predicted:")
    println(mMetrics.confusionMatrix)

    labels.foreach { l =>
      println(s"Precision($l) = " + mMetrics.precision(l))
    }

    labels.foreach { l =>
      println(s"Recall($l) = " + mMetrics.recall(l))
    }

    labels.foreach { l =>
      println(s"F1-Score($l) = " + mMetrics.fMeasure(l))
    }

  }
}