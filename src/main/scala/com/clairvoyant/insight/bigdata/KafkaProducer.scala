package com.clairvoyant.insight.bigdata

import java.util.Properties

import com.typesafe.config.{Config, ConfigFactory}
import org.apache.kafka.clients.producer.{KafkaProducer, ProducerRecord}
import scala.util.Random

object KafkaProducer {

  def main(args: Array[String]): Unit = {

    // Load values form the Config file(application.json)
    val config: Config = ConfigFactory.load("application.json")

    val KAFKA_TOPIC: String = config.getString("kafka.producer_topic")
    val KAFKA_BROKERS: String = config.getString("kafka.brokers")

    val  props = new Properties()
    props.put("bootstrap.servers", KAFKA_BROKERS)

    props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer")
    props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer")

    val producer = new KafkaProducer[String, String](props)

    val TOPIC= KAFKA_TOPIC

    val SetosaSepalLengthsRange = Set(4.95,4.96,4.97,4.98,4.99)
    val SetosaPetalLengthsRange = Set(1.43,1.44,1.45,1.46,1.47)
    val SetosaSepalWidthsRange = Set(3.37,3.38,3.39,3.40,3.41)
    val SetosaPetalWidthsRange = Set(0.21,0.22,0.23,0.24,0.25)

    val VersicolorSepalLengthsRange = Set(5.92,5.93,5.94,5.95,5.96)
    val VersicolorPetalLengthsRange = Set(4.25,4.26,4.27,4.28,4.29)
    val VersicolorSepalWidthsRange = Set(2.75,2.76,2.77,2.78,2.79)
    val VersicolorPetalWidthsRange = Set(1.32,1.33,1.34,1.35,1.36)

    val VirginicaSepalLengthsRange = Set(6.62,6.63,6.64,6.65,6.66)
    val VirginicaPetalLengthsRange = Set(5.59,5.60,5.61,5.62,5.63)
    val VirginicaSepalWidthsRange = Set(2.99,3.0,3.01,3.02,3.03)
    val VirginicaPetalWidthsRange = Set(2.02,2.03,2.04,2.05,2.06)

    while(true){
      val rnd = new Random

      val SetosaSepalLength = SetosaSepalLengthsRange.toVector(rnd.nextInt(SetosaSepalLengthsRange.size))
      val SetosaPetalLength = SetosaPetalLengthsRange.toVector(rnd.nextInt(SetosaPetalLengthsRange.size))
      val SetosaSepalWidth = SetosaSepalWidthsRange.toVector(rnd.nextInt(SetosaSepalWidthsRange.size))
      val SetosaPetalWidth = SetosaPetalWidthsRange.toVector(rnd.nextInt(SetosaPetalWidthsRange.size))

      val VersicolorSepalLength = VersicolorSepalLengthsRange.toVector(rnd.nextInt(SetosaSepalLengthsRange.size))
      val VersicolorPetalLength = VersicolorPetalLengthsRange.toVector(rnd.nextInt(SetosaPetalLengthsRange.size))
      val VersicolorSepalWidth = VersicolorSepalWidthsRange.toVector(rnd.nextInt(SetosaSepalWidthsRange.size))
      val VersicolorPetalWidth = VersicolorPetalWidthsRange.toVector(rnd.nextInt(SetosaPetalWidthsRange.size))

      val VirginicaSepalLength = VirginicaSepalLengthsRange.toVector(rnd.nextInt(SetosaSepalLengthsRange.size))
      val VirginicaPetalLength = VirginicaPetalLengthsRange.toVector(rnd.nextInt(SetosaPetalLengthsRange.size))
      val VirginicaSepalWidth = VirginicaSepalWidthsRange.toVector(rnd.nextInt(SetosaSepalWidthsRange.size))
      val VirginicaPetalWidth = VirginicaPetalWidthsRange.toVector(rnd.nextInt(SetosaPetalWidthsRange.size))

      val SetosaKafkaRecord = "{\"SepalLength\":" + SetosaSepalLength + " ,\"PetalLength\":" + SetosaPetalLength + ",\"SepalWidth\":" + SetosaSepalWidth + ",\"PetalWidth\":" + SetosaPetalWidth + ",\"flower\":\"setosa\"}"
      val setosa_record = new ProducerRecord(TOPIC, "key", SetosaKafkaRecord)
      producer.send(setosa_record)
      println("Pushed Setosa Record: " + SetosaKafkaRecord)

      val VersicolorKafkaRecord = "{\"SepalLength\":" + VersicolorSepalLength + " ,\"PetalLength\":" + VersicolorPetalLength + ",\"SepalWidth\":" + VersicolorSepalWidth + ",\"PetalWidth\":" + VersicolorPetalWidth + ",\"flower\":\"versicolor\"}"
      val versicolor_record = new ProducerRecord(TOPIC, "key", VersicolorKafkaRecord)
      producer.send(versicolor_record)
      println("Pushed Versicolor Record: " + VersicolorKafkaRecord)

      val VirginicaKafkaRecord = "{\"SepalLength\":" + VirginicaSepalLength + " ,\"PetalLength\":" + VirginicaPetalLength + ",\"SepalWidth\":" + VirginicaSepalWidth + ",\"PetalWidth\":" + VirginicaPetalWidth + ",\"flower\":\"virginica\"}"
      val virginica_record = new ProducerRecord(TOPIC, "key", VirginicaKafkaRecord)
      producer.send(virginica_record)
      println("Pushed Virginica Record: " + VirginicaKafkaRecord)

      Thread.sleep(10000)
    }
    producer.close()

  }

}
