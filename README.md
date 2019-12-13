# SparkML
## 1. Add dependencies
Append to file build.sbt
```
libraryDependencies += "org.apache.spark" %% "spark-core" % "2.4.4"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.4.4"
```
## 2. Demo
Run the demo code at src/main/scala/begin/CancerPrediction.scala

## 3. Main steps

- Create spark Session with n processes:
```
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().master("local[n]").getOrCreate()
```
- Load data:
```
val df = spark.read.option("header", "true")
 .option("inferSchema", "true")
 .option("delimiter", ",").csv("file.csv")
```
- Check schema:
```
df.printSchema()
```
- Set log type to ERROR:
```
import org.apache.log4j
log4j.Logger.getLogger("org").setLevel(log4j.Level.ERROR)
```

- Replace None value by median:
```
val col1_median_array = df.stat.approxQuantile("col1", Array(0.5), 0)
val col1_median = col1_median_array(0)
val col2_median_array = df.stat.approxQuantile("col2", Array(0.5), 0)
val col2_median = col2_median_array(0)
val filled = df.na.fill(Map("col1" -> col1_median,"col2" -> col2_median))
```

- Convert nominal values to numerical values:
```
import org.apache.spark.ml.feature.StringIndexer
val df2 = new StringIndexer()
            .setInputCol("input_col")
            .setOutputCol("output_col")
            .setHandleInvalid("keep")
            .fit(df).transform(df)
```

- Generate features vector:

```
import org.apache.spark.ml.feature.VectorAssembler
val assembler = new VectorAssembler()
            .setInputCols(Array("col1", "col2", "col3"))
            .setOutputCol("features")
val output = assembler.transform(df).select("label", "features")
```
- Split data into trainset and testset:
```
val Array(train, test) = df.select("label","features")
                        .randomSplit(Array(0.7, 0.3), seed = 1)
```
- Train model:
```
import org.apache.spark.ml.classification.DecisionTreeClassifier
val rf = new DecisionTreeClassifier()
val model = rf.fit(training)
```
- Get result:
```
val results = model.transform(test).select("prediction", "label")
val predictionAndLabels = results.as[(Double, Double)].rdd
```

- Get the RMSE:
```
import org.apache.spark.mllib.evaluation.RegressionMetrics
val regressionMetrics = new RegressionMetrics(predictionAndLabels)
println(s"RMSE with RegressionMetrics: ${regressionMetrics.rootMeanSquaredError}")
```
- Create evaluation:
```
import org.apache.spark.mllib.evaluation.MulticlassMetrics
val mMetrics = new MulticlassMetrics(predictionAndLabels)
```
- Get accuracy and confusion matrix:
```
println("Accuracy: " + mMetrics.accuracy)
println("Confusion matrix:")
println(mMetrics.confusionMatrix)
```
- Get the Precision, Recall and F1 score:
```
labels.foreach { l =>
println(s"Precision($l) = " + mMetrics.precision(l))
}

labels.foreach { l =>
println(s"Recall($l) = " + mMetrics.recall(l))
}

labels.foreach { l =>
println(s"F1-Score($l) = " + mMetrics.fMeasure(l))
}
```

## 4. Exercise

Using HarvardX-MITx Person-Course Academic dataset (src/main/scala/begin/mooc.scala)
to predict whether or not the participant earned a certificate.
- label: "certified" column 
- features: "registered", "viewed",  "explored", "final_cc_cname_DI", "gender", "nevents",
 "ndays_act", "nplay_video", "nchapters", "nforum_posts" columns