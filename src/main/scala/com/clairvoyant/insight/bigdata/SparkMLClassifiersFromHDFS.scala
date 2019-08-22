package com.clairvoyant.insight.bigdata

import com.typesafe.config.{Config, ConfigFactory}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassifier, LogisticRegression, NaiveBayes, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.udf
import org.slf4j.{Logger, LoggerFactory}

object SparkMLClassifiersFromHDFS {

    def main(args: Array[String]): Unit = {

        val LOGGER: Logger = LoggerFactory.getLogger(SparkMLClassifiersFromHDFS.getClass.getName)

        // Load values form the Config file(application.json)
        val config: Config = ConfigFactory.load("application.json")

        val SPARK_MASTER = config.getString("spark.master")
        val SPARK_APP_NAME = config.getString("ml.app_name")
        val DATASET_PATH = config.getString("ml.dataset_path")
        val ML_CLASSIFIER = config.getString("ml.classifier")

        val spark = SparkSession
                .builder
                .master(SPARK_MASTER)
                .appName(SPARK_APP_NAME)
                .getOrCreate()

        val data = spark.read.format("csv").option("header", "true").load(DATASET_PATH)

        val toDouble = udf[Double, String](_.toDouble)

        val df = data.withColumn("PetalLength", toDouble(data("PetalLength"))).withColumn("PetalWidth", toDouble(data("PetalWidth"))).withColumn("SepalLength", toDouble(data("SepalLength"))).withColumn("SepalWidth", toDouble(data("SepalWidth")))

        val assembler = new VectorAssembler()
                .setInputCols(Array("PetalLength", "PetalWidth", "SepalLength", "SepalWidth"))
                .setOutputCol("features")

        val transformed_data = assembler.transform(df).drop("PetalLength", "PetalWidth", "SepalLength", "SepalWidth")

        transformed_data.show()
        transformed_data.printSchema()

        val labelIndexer = new StringIndexer()
                .setInputCol("flower")
                .setOutputCol("indexedFlower")
                .fit(transformed_data)

        // Automatically identify categorical features, and index them.
        val featureIndexer = new VectorIndexer()
                .setInputCol("features")
                .setOutputCol("indexedFeatures")
                .setMaxCategories(4)
                .fit(transformed_data)

        // Split the data into training and test sets (30% held out for testing).
        val Array(trainingData, testData) = transformed_data.randomSplit(Array(0.7, 0.3))

        testData.show()

        // Convert indexed labels back to original labels.
        val labelConverter = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(labelIndexer.labels)

        val pipeline = new Pipeline()

        // TRAIN THE MODEL WITH RESPECTIVE ML_CLASSIFIER
        if (ML_CLASSIFIER == "DecisionTree") {

            LOGGER.info("Training DecisionTree Model")

            // Train a DecisionTree
            val dt: DecisionTreeClassifier = new DecisionTreeClassifier()
                    .setLabelCol("indexedFlower")
                    .setFeaturesCol("indexedFeatures")

            pipeline.setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

            LOGGER.info("Finished Training DecisionTree Model")

        } else if (ML_CLASSIFIER == "LogisticRegression") {

            LOGGER.info("Training LogisticRegression Model")

            // Train the Logistic Regression.
            val lr: LogisticRegression = new LogisticRegression()
                    .setLabelCol("indexedFlower")
                    .setMaxIter(10)
                    .setRegParam(0.3)
                    .setElasticNetParam(0.8)
                    .setFamily("multinomial")

            pipeline.setStages(Array(labelIndexer, featureIndexer, lr, labelConverter))

            LOGGER.info("Finished Training LogisticRegression Model")

        } else if (ML_CLASSIFIER == "NaiveBayes") {

            LOGGER.info("Training NaiveBayes Model")

            // Train the Naive Bayes
            val nb: NaiveBayes = new NaiveBayes().setLabelCol("indexedFlower")

            pipeline.setStages(Array(labelIndexer, featureIndexer, nb, labelConverter))

            LOGGER.info("Finished Training NaiveBayes Model")

        } else if (ML_CLASSIFIER == "RandomForest") {

            LOGGER.info("Training RandomForest Model")

            // Train a RandomForest
            val rf: RandomForestClassifier = new RandomForestClassifier()
                    .setLabelCol("indexedFlower")
                    .setFeaturesCol("indexedFeatures")
                    .setNumTrees(10)

            pipeline.setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

            LOGGER.info("Finished Training RandomForest Model")


        } else {
            LOGGER.error("Provide Correct Classifier Name as Given Below")
            LOGGER.error("1. DecisionTree")
            LOGGER.error("2. LogisticRegression")
            LOGGER.error("3. NaiveBayes")
            LOGGER.error("4. RandomForest")
            System.exit(0)
        }

        // Train model. This also runs the indexers.
        val model = pipeline.fit(trainingData)

        // Make predictions.
        val predictions = model.transform(testData)

        predictions.show()
        LOGGER.info("Count: " + predictions.count().toString)

        val evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("indexedFlower")
                .setPredictionCol("prediction")
                .setMetricName("accuracy")
        val accuracy = evaluator.evaluate(predictions)

        println("Accuracy: " + accuracy)
        println("Test Error = " + (1.0 - accuracy))


        //    predictions.filter("predictedLabel != 'setosa'").show()

        spark.stop()

    }

}
