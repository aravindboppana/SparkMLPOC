package com.clairvoyant.insight.bigdata

import com.typesafe.config.{Config, ConfigFactory}
import org.apache.kafka.common.serialization.StringDeserializer
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.streaming.kafka010.{ConsumerStrategies, KafkaUtils}
import org.apache.spark.streaming.kafka010.LocationStrategies.PreferConsistent
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.slf4j.{Logger, LoggerFactory}

object SparkStreamingToHDFS {

    def main(args: Array[String]): Unit = {

        // Load values form the Config file(application.json)
        val config: Config = ConfigFactory.load("application.json")

        val SPARK_APP_NAME: String = config.getString("spark.app_name")
        val SPARK_MASTER: String = config.getString("spark.master")
        val SPARK_BATCH_DURATION: Int = config.getInt("spark.batch_duration")

        val KAFKA_TOPICS: String = config.getString("kafka.consumer_topic")
        val KAFKA_BROKERS: String = config.getString("kafka.brokers")
        val KAFKA_GROUP_ID: String = config.getString("kafka.group_id")
        val KAFKA_OFFSET_RESET: String = config.getString("kafka.auto_offset_reset")

        val HDFS_STORAGE_LOCATION: String = config.getString("hdfs.storage_location")

        val LOGGER: Logger = LoggerFactory.getLogger(SparkStreamingToHDFS.getClass.getName)

        val sparkConf = new SparkConf().setAppName(SPARK_APP_NAME).setMaster(SPARK_MASTER)
        val sparkStreamingContext = new StreamingContext(sparkConf, Seconds(SPARK_BATCH_DURATION))

        val spark = SparkSession.builder.config(sparkConf).getOrCreate()
        import spark.implicits._

        val kafkaParams = Map[String, Object]("bootstrap.servers" -> KAFKA_BROKERS,
            "key.deserializer" -> classOf[StringDeserializer],
            "value.deserializer" -> classOf[StringDeserializer],
            "group.id" -> KAFKA_GROUP_ID,
            "auto.offset.reset" -> KAFKA_OFFSET_RESET,
            "enable.auto.commit" -> (false: java.lang.Boolean))

        val topicsSet = KAFKA_TOPICS.split(",").toSet

        val kafkaStream = KafkaUtils.createDirectStream[String, String](sparkStreamingContext, PreferConsistent, ConsumerStrategies.Subscribe[String, String](topicsSet, kafkaParams))

        kafkaStream.map(message => {
            message.value().toString
        }).foreachRDD(rdd => {
            if (!rdd.isEmpty()) {

                val df = spark.read.json(rdd.toDS())

                df.printSchema()
                df.show()

                df.write.mode("append").option("header", "true").csv(HDFS_STORAGE_LOCATION)

            }
        })

        sparkStreamingContext.start()
        sparkStreamingContext.awaitTermination()


    }

}
