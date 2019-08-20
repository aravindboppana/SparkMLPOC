package com.clairvoyant.insight.bigdata

import com.typesafe.config.{Config, ConfigFactory}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.udf

object SparkMLFromHDFSNaiveBayesClassifier {

  def main(args: Array[String]): Unit = {

    // Load values form the Config file(application.json)
    val config: Config = ConfigFactory.load("application.json")

    val SPARK_MASTER: String = config.getString("spark.master")
    val SPARK_APP_NAME = "SparkMLRandomForestClassifier"

    val spark = SparkSession
      .builder
      .master(SPARK_MASTER)
      .appName(SPARK_APP_NAME)
      .getOrCreate()

    val data = spark.read.format("csv").option("header", "true").load("/Users/aravind/Projects/SparkML/flower_dataset.csv")

    val toDouble = udf[Double, String](_.toDouble)

    val df2 = data.withColumn("PetalLength", toDouble(data("PetalLength"))).withColumn("PetalWidth", toDouble(data("PetalWidth"))).withColumn("SepalLength", toDouble(data("SepalLength"))).withColumn("SepalWidth", toDouble(data("SepalWidth")))

    val assembler = new VectorAssembler()
      .setInputCols(Array("PetalLength", "PetalWidth", "SepalLength", "SepalWidth"))
      .setOutputCol("features")

    val outdata = assembler.transform(df2).drop("PetalLength", "PetalWidth", "SepalLength", "SepalWidth")

    outdata.show()
    outdata.printSchema()

    val labelIndexer = new StringIndexer()
      .setInputCol("flower")
      .setOutputCol("indexedFlower")
      .fit(outdata)

    // Automatically identify categorical features, and index them.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(outdata)

    val Array(trainingData, testData) = outdata.randomSplit(Array(0.7, 0.3))

    // Train a RandomForest model.
    val nbClassifier = new NaiveBayes()

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Chain indexers and forest in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, nbClassifier, labelConverter))

    // Train model. This also runs the indexers.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    // Select example rows to display.
    predictions.show()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedFlower")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)

    println("Accuracy: " + accuracy)
    println("Test Error = " + (1.0 - accuracy))

    spark.stop()


  }

}
